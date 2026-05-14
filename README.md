# stego_mom_polar

## 项目说明

**stego_mom_polar** 是一套**生成式图像隐写**实验代码：用 VQGAN 从密文图像得到比特载荷，在 SD 2.1 潜空间经**正交映射**与**极化码**（多通道）编码后生成载密图像；接收端对受攻击图像做扩散反演，用**矩估计（MoM）**得到各通道 LLR 再极化译码恢复比特并重建成图。

a minimal runnable pipeline for **orthogonal latent mapping + polar coding** under **Stable Diffusion 2.1**, with **MoM-based** soft LLRs for decoding. 

---

## 项目声明 · Project Statement

**本项目的作者及单位** · The author and affiliation of this project:
| Field · 字段 | Content · 内容 |
| :--- | :--- |
| **项目名称** *Project Name* | `stego_mom_polar`（正交映射 + 极化码潜空间隐写） |
| **项目作者** *Authors* | DuanHaohan |
| **作者单位** *Affiliation* | 暨南大学网络空间安全学院 · College of Cyber Security, Jinan University |

