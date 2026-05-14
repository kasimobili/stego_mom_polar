#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import glob
import os
import shutil
import cv2
import numpy as np
import skimage
import io
import torch
from PIL import Image
from torchvision.transforms.functional import normalize
from edict_functions import coupled_stablediffusion
import utils
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.misc import get_device
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.metrics import calculate_psnr, calculate_ssim
from polarcodes import PolarCode, Construct, Encode, Decode
from robust_codebook_indices import RUBUST_DECIMAL_INDEX

class OrthogonalMapper:
    """Map (4, 4096) binary codewords to (1,4,64,64) latents via Z = C M C^T per channel, then per-channel norm."""

    def __init__(self, size: int = 64, seed: int = 10086, device: str = "cuda"):
        self.size = size
        self.device = device
        torch.manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(size, size, device=device))
        if torch.det(q) < 0:
            q[:, -1] *= -1
        self.C = q.to(device)
        self.C_T = self.C.t()

    def forward(self, binary_data: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(binary_data, np.ndarray):
            binary_data = torch.from_numpy(binary_data).float().to(self.device)
        if binary_data.dim() == 1:
            binary_data = binary_data.view(4, 4096)
        assert binary_data.shape == (4, 4096), binary_data.shape
        m = binary_data.view(4, self.size, self.size)
        m_sym = 2.0 * m - 1.0
        inter = torch.bmm(m_sym, self.C_T.unsqueeze(0).expand(4, -1, -1))
        z = torch.bmm(self.C.unsqueeze(0).expand(4, -1, -1), inter)
        z_mean = z.mean(dim=(1, 2), keepdim=True)
        z_std = z.std(dim=(1, 2), keepdim=True) + 1e-8
        return ((z - z_mean) / z_std).unsqueeze(0)

    def backward(self, latents: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).float().to(self.device)
        z = latents.squeeze(0).float() if latents.dim() == 4 else latents.float()
        c_tb = self.C_T.unsqueeze(0).expand(4, -1, -1)
        inter = torch.bmm(c_tb, z)
        m_p = torch.bmm(inter, self.C.unsqueeze(0).expand(4, -1, -1))
        return m_p.view(4, 4096)


# ---------------------------------------------------------------------------
# MoM -> polar channel LLRs (per test_alpha; no SNR aggregation)
# ---------------------------------------------------------------------------


def soft_obs_to_polar_llrs(soft_channel: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    From inverse-orthogonal soft observations hat{m}_v, estimate alpha, sigma^2 via 2nd/4th moments
    and return clipped polar LLRs for one (4096,) channel.
    """
    hat_mv = np.asarray(soft_channel, dtype=np.float64)
    m_2 = float(np.mean(hat_mv ** 2))
    m_4 = float(np.mean(hat_mv ** 4))
    term = max((3.0 * m_2 ** 2 - m_4) / 2.0, 1e-12)
    alpha = term ** 0.25
    sigma2 = max(m_2 - alpha ** 2, 1e-8)
    llrs = -1.0 * (2.0 * hat_mv) * alpha / sigma2
    return llrs, alpha, sigma2


# ---------------------------------------------------------------------------
# Steganography pipeline
# ---------------------------------------------------------------------------


class StegoOrchestrator:
    """Polar (4096,640) x4 + orthogonal map + SD2.1; decode with MoM LLR (no fixed sigma2)."""

    K, N, num_channels, total_secret_bits = 640, 4096, 4, 2560

    def __init__(self, device: str = "cuda", snr_db: float = 2.5):
        self.device = device
        self.polar_coders: list[PolarCode] = []
        for _ in range(self.num_channels):
            pc = PolarCode(self.N, self.K)
            pc.construction_type = "bb"
            Construct(pc, snr_db)
            self.polar_coders.append(pc)
        self.mapper = OrthogonalMapper(size=64, seed=10086, device=device)

    def run_experiment(
        self,
        secret_bits_2560: np.ndarray | list | torch.Tensor,
        prompt: str = "a beautiful landscape",
        attack_type: str | None = None,
        attack_params: dict | None = None,
        skip_recovery: bool = False,
    ) -> dict:
        if isinstance(secret_bits_2560, list):
            secret_bits_2560 = np.array(secret_bits_2560, dtype=np.uint8)
        elif isinstance(secret_bits_2560, torch.Tensor):
            secret_bits_2560 = secret_bits_2560.cpu().numpy()
        assert len(secret_bits_2560) == self.total_secret_bits

        chunks = [secret_bits_2560[i * self.K : (i + 1) * self.K] for i in range(self.num_channels)]
        encoded_chunks = []
        for chunk, pc in zip(chunks, self.polar_coders):
            pc.set_message(chunk)
            Encode(pc)
            encoded_chunks.append(pc.get_codeword())
        encoded_data = np.stack(encoded_chunks, axis=0)

        z_t = self.mapper.forward(encoded_data)
        gen_out, _, _ = coupled_stablediffusion(
            prompt=prompt,
            reverse=False,
            run_baseline=True,
            guidance_scale=7.5,
            steps=50,
            init_noise=z_t,
        )
        stego_image = gen_out[0] if isinstance(gen_out, list) else gen_out

        if attack_type is not None and attack_params is not None:
            attacked_image = self._apply_attack(stego_image, attack_type, attack_params)
        else:
            attacked_image = stego_image

        if skip_recovery:
            return {
                "ber": 0.0,
                "bit_acc": 0.0,
                "bit_errors": 0,
                "total_bits": 0,
                "recovered_bits": None,
                "original_bits": None,
                "stego_image": stego_image,
                "attacked_image": attacked_image,
            }

        init_noise2, _ = coupled_stablediffusion(
            prompt="",
            reverse=True,
            run_baseline=True,
            guidance_scale=1,
            init_image=attacked_image,
        )
        z_t_prime = init_noise2[0]
        soft_llrs = self.mapper.backward(z_t_prime)

        recovered_chunks = []
        alphas: list[float] = []
        sigma2s: list[float] = []
        for i, pc in enumerate(self.polar_coders):
            llrs, alpha, sigma2 = soft_obs_to_polar_llrs(soft_llrs[i].cpu().numpy())
            alphas.append(alpha)
            sigma2s.append(sigma2)
            pc.likelihoods = llrs
            Decode(pc)
            recovered_chunks.append(np.asarray(pc.message_received[: self.K], dtype=np.uint8))

        recovered_bits = np.concatenate(recovered_chunks)
        bit_acc_per_ch = []
        for i in range(self.num_channels):
            o = secret_bits_2560[i * self.K : (i + 1) * self.K]
            r = recovered_chunks[i]
            err = int(np.sum(o != r))
            bit_acc_per_ch.append(1.0 - err / self.K)

        min_len = min(len(secret_bits_2560), len(recovered_bits))
        bit_errors = int(np.sum(secret_bits_2560[:min_len] != recovered_bits[:min_len]))
        ber = bit_errors / min_len if min_len else 1.0

        return {
            "ber": ber,
            "bit_acc": 1.0 - ber,
            "bit_acc_per_ch": bit_acc_per_ch,
            "bit_errors": bit_errors,
            "total_bits": min_len,
            "recovered_bits": recovered_bits,
            "original_bits": secret_bits_2560[:min_len],
            "stego_image": stego_image,
            "attacked_image": attacked_image,
            "alpha": float(np.mean(alphas)),
            "sigma2": float(np.mean(sigma2s)),
        }

    def extract_from_attacked_image(self, attacked_image, secret_bits_2560: np.ndarray) -> dict:
        init_noise2, _ = coupled_stablediffusion(
            prompt="",
            reverse=True,
            run_baseline=True,
            guidance_scale=1,
            init_image=attacked_image,
        )
        soft_llrs = self.mapper.backward(init_noise2[0])
        recovered_chunks = []
        alphas: list[float] = []
        sigma2s: list[float] = []
        for i, pc in enumerate(self.polar_coders):
            llrs, alpha, sigma2 = soft_obs_to_polar_llrs(soft_llrs[i].cpu().numpy())
            alphas.append(alpha)
            sigma2s.append(sigma2)
            pc.likelihoods = llrs
            Decode(pc)
            recovered_chunks.append(np.asarray(pc.message_received[: self.K], dtype=np.uint8))

        recovered_bits = np.concatenate(recovered_chunks)
        bit_acc_per_ch = []
        for i in range(self.num_channels):
            o = secret_bits_2560[i * self.K : (i + 1) * self.K]
            r = recovered_chunks[i]
            err = int(np.sum(o != r))
            bit_acc_per_ch.append(1.0 - err / self.K)

        min_len = min(len(secret_bits_2560), len(recovered_bits))
        bit_errors = int(np.sum(secret_bits_2560[:min_len] != recovered_bits[:min_len]))
        ber = bit_errors / min_len if min_len else 1.0

        return {
            "ber": ber,
            "bit_acc": 1.0 - ber,
            "bit_acc_per_ch": bit_acc_per_ch,
            "bit_errors": bit_errors,
            "total_bits": min_len,
            "recovered_bits": recovered_bits,
            "alpha": float(np.mean(alphas)),
            "sigma2": float(np.mean(sigma2s)),
        }

    def _apply_attack(self, image, attack_type: str, attack_params: dict) -> Image.Image:
        img_np = np.array(image) if isinstance(image, Image.Image) else image
        if attack_type == "jpeg":
            buf = io.BytesIO()
            (image if isinstance(image, Image.Image) else Image.fromarray(img_np)).save(
                buf, format="JPEG", quality=attack_params.get("jpeg_qf", 30)
            )
            buf.seek(0)
            return Image.open(buf)
        if attack_type == "gaussian":
            var = attack_params.get("gaussian_var", 0.035) ** 2
            noisy = skimage.util.random_noise(img_np, mode="gaussian", mean=0, var=var)
            return Image.fromarray(np.uint8(noisy * 255))
        if attack_type == "median":
            k = attack_params.get("median_k", 15)
            return Image.fromarray(cv2.medianBlur(img_np, k))
        if attack_type == "gauss_blur":
            k = attack_params.get("gauss_blur_k", 15)
            return Image.fromarray(cv2.GaussianBlur(img_np, (k, k), 0))
        if attack_type == "saltandpepper":
            amt = attack_params.get("salt_pepper_amount", 0.05)
            noisy = skimage.util.random_noise(img_np, mode="s&p", amount=amt)
            return Image.fromarray(np.uint8(noisy * 255))
        if attack_type == "resize":
            scale = attack_params.get("resize_scale", 0.8)
            pil = image if isinstance(image, Image.Image) else Image.fromarray(img_np)
            w, h = pil.size
            small = pil.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            return small.resize((w, h), Image.BILINEAR)
        return image if isinstance(image, Image.Image) else Image.fromarray(img_np)


def load_vqgan_with_robust_codebook(device: str, ckpt_path: str = "./weights/vqgan_code1024.pth"):
    vqgan = ARCH_REGISTRY.get("VQAutoEncoder")(
        img_size=512,
        nf=64,
        ch_mult=[1, 2, 2, 4, 4, 8],
        quantizer="nearest",
        res_blocks=2,
        attn_resolutions=[16],
        codebook_size=1024,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)["params_ema"]
    vqgan.load_state_dict(ckpt)
    vqgan.eval()
    codebook = torch.Tensor(np.array(vqgan.quantize.embedding.weight.tolist()))
    new_cb = torch.zeros(1024, 256)
    for i, idx in enumerate(RUBUST_DECIMAL_INDEX):
        new_cb[idx] = codebook[i]
    vqgan.quantize.embedding.weight = torch.nn.Parameter(new_cb.to(device))
    return vqgan


def _collect_images(input_path: str, max_images: int) -> list[str]:
    if input_path.lower().endswith((".jpg", ".jpeg", ".png")):
        return [input_path]
    paths = sorted(glob.glob(os.path.join(input_path, "*.[jpJP][pnPN]*[gG]")))
    return paths[:max_images]


def main() -> None:
    device = get_device()
    parser = argparse.ArgumentParser(description="MoM polar + orthogonal latent stego (test_alpha logic)")
    parser.add_argument("-i", "--input_path", type=str, default="./images", help="Image file or directory")
    parser.add_argument("-n", "--max_images", type=int, default=20, help="Max images when input is a directory")
    parser.add_argument("--snr_db", type=float, default=2.5, help="Polar construction SNR (dB) for Construct()")
    args = parser.parse_args()

    vqgan = load_vqgan_with_robust_codebook(device)

    jpeg_qf = 90
    gaussian_var = 0.005
    median_k = 3
    gaussian_blur_k = 3
    salt_pepper_amount = 0.01
    resize_scale = 0.9

    results_root = f"results_mom_{jpeg_qf}_{gaussian_var}_{median_k}_{gaussian_blur_k}/"
    if os.path.exists(results_root):
        shutil.rmtree(results_root)
    os.makedirs(results_root, exist_ok=True)

    attack_configs = [
        ("jpeg", {"jpeg_qf": jpeg_qf}),
        ("gaussian", {"gaussian_var": gaussian_var}),
        ("median", {"median_k": median_k}),
        ("gauss_blur", {"gauss_blur_k": gaussian_blur_k}),
        ("saltandpepper", {"salt_pepper_amount": salt_pepper_amount}),
        ("resize", {"resize_scale": resize_scale}),
        (None, None),
    ]
    atk_names = [
        f"jpeg_{jpeg_qf}",
        f"gaussian_{gaussian_var}",
        f"median_{median_k}",
        f"gauss_blur_{gaussian_blur_k}",
        f"saltandpepper_{salt_pepper_amount}",
        f"resize_{resize_scale}",
        "origin",
    ]
    num_attacks = len(atk_names)
    psnr_sum = [0.0] * num_attacks
    ssim_sum = [0.0] * num_attacks

    img_list = _collect_images(args.input_path, args.max_images)
    prompt = "highly detailed concept art of a sakura plum tree made with water, overgrowth, Makoto Shinkai"

    for i, img_path in enumerate(img_list):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        result_dir = os.path.join(results_root, basename)
        os.makedirs(result_dir, exist_ok=True)

        secret_img = cv2.imread(img_path)
        secret_t = img2tensor(secret_img / 255.0, bgr2rgb=True, float32=True)
        normalize(secret_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        secret_t = secret_t.unsqueeze(0).to(device)

        txt_path = os.path.join(result_dir, f"{basename}_info.txt")
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            with torch.no_grad():
                indice = vqgan.get_indice(vqgan.VQEncoder(secret_t)).tolist()
                bits = "".join(format(x, "010b") for x in indice)
                binary_array = np.array([int(b) for b in bits], dtype=np.uint8)

                orch = StegoOrchestrator(device=device, snr_db=args.snr_db)
                origin = orch.run_experiment(binary_array, prompt=prompt, attack_type=None, attack_params=None)
                stego_img = origin["stego_image"]

                stego_bgr = cv2.cvtColor(np.asarray(stego_img), cv2.COLOR_RGB2BGR)
                imwrite(stego_bgr, os.path.join(result_dir, f"{basename}_stego.png"))
                imwrite(secret_img, os.path.join(result_dir, f"{basename}_secret.png"))

                def log_origin(tag: str, res: dict) -> None:
                    print(f"{tag} BER={res['ber']:.6f} bit_acc={res['bit_acc']:.6f} alpha={res['alpha']:.4f} sigma2={res['sigma2']:.6f}")
                    print(f"{tag} BER={res['ber']:.6f} bit_acc={res['bit_acc']:.6f}", file=txt_file)

                log_origin("Origin", origin)

                rec_list = origin["recovered_bits"][:2560].tolist()
                if len(rec_list) < 2560:
                    rec_list += [0] * (2560 - len(rec_list))
                avg_idx = utils.convert_binary_to_decimal(rec_list, 10)
                recon = tensor2img(vqgan.VQDecoder(vqgan.get_quant_feat(avg_idx).to(device)), rgb2bgr=True, min_max=(-1, 1))
                imwrite(recon, os.path.join(result_dir, f"{basename}_recon_origin.png"))
                psnr_o = calculate_psnr(recon, secret_img, 0)
                ssim_o = calculate_ssim(recon, secret_img, 0)
                psnr_sum[6] += psnr_o
                ssim_sum[6] += ssim_o
                print(f"Origin PSNR={psnr_o:.2f} SSIM={ssim_o:.4f}", file=txt_file)

                for idx_atk, (atk_t, atk_p) in enumerate(attack_configs):
                    if atk_t is None:
                        continue
                    name = atk_names[idx_atk]
                    attacked = orch._apply_attack(stego_img, atk_t, atk_p)
                    res = orch.extract_from_attacked_image(attacked, binary_array)
                    rec = res["recovered_bits"]
                    rec_list = rec[:2560].tolist() if len(rec) >= 2560 else rec.tolist()
                    if len(rec_list) < 2560:
                        rec_list += [0] * (2560 - len(rec_list))
                    avg_idx = utils.convert_binary_to_decimal(rec_list, 10)
                    recon = tensor2img(vqgan.VQDecoder(vqgan.get_quant_feat(avg_idx).to(device)), rgb2bgr=True, min_max=(-1, 1))
                    imwrite(recon, os.path.join(result_dir, f"{basename}_recon_{name}.png"))
                    psnr_v = calculate_psnr(recon, secret_img, 0)
                    ssim_v = calculate_ssim(recon, secret_img, 0)
                    psnr_sum[idx_atk] += psnr_v
                    ssim_sum[idx_atk] += ssim_v
                    print(f"{name} BER={res['ber']:.6f} PSNR={psnr_v:.2f} SSIM={ssim_v:.4f}", file=txt_file)

                n_done = i + 1
                psnr_avg = [p / n_done for p in psnr_sum]
                ssim_avg = [s / n_done for s in ssim_sum]
                with open(os.path.join(results_root, "summary.txt"), "w", encoding="utf-8") as sf:
                    sf.write(f"images={n_done}\nAvg PSNR: {psnr_avg}\nAvg SSIM: {ssim_avg}\n")
                print(f"[{n_done}/{len(img_list)}] {basename} done. Avg PSNR tail={psnr_avg[-1]:.2f}")


if __name__ == "__main__":
    main()
