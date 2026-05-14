import torch
import random
import torch.nn.functional as F

# 用于鲁棒性步骤2，返回一个数0-1023翻转一位的10个数
def flip_bits(decimal):
    # 将十进制整数转换为10比特的二进制字符串
    binary = format(decimal, '010b')

    # 对每个比特进行翻转，并转换回十进制整数
    flipped_decimals = []
    for i in range(10):
        flipped_binary = binary[:i] + str(1 - int(binary[i])) + binary[i + 1:]
        flipped_decimal = int(flipped_binary, 2)
        flipped_decimals.append(flipped_decimal)

    return flipped_decimals


# 输入长度256的indices list，返回一个1024的bit list，出现过的indices在 1024对应序列中的比特为1
def indices_flag_1024(indices):
    indices_flag = [0] * 1024
    for ind in indices:
        indices_flag[ind] = 1
    return indices_flag


def find_most_similar_indices(vectors):
    # 计算每个向量与其他向量的余弦相似度
    similarities = F.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=2)
    # 将对角线上的相似度设置为负无穷大，排除与自身的相似度
    similarities.fill_diagonal_(-float('inf'))
    # 找到每个向量对应的余弦相似度最大的索引
    most_similar_indices = torch.argmax(similarities, dim=1)

    sorted_tensor, sorted_indices = torch.sort(similarities, dim=1)

    # 找到每个向量对应的欧几里得距离最小的索引
    euc_distance = torch.cdist(vectors, vectors)
    euc_distance.fill_diagonal_(float('inf'))
    most_similar_indices1 = torch.argmin(euc_distance, dim=1)

    sorted_tensor1, sorted_indices1 = torch.sort(euc_distance, dim=1)

    return most_similar_indices1


# 找表中最相似两个向量之间indice的汉明距离
def hamming_distance_list(lst):
    hamming_distances = []
    for i, num in enumerate(lst):
        binary_num = bin(num)[2:].zfill(10)  # 转换为10位的二进制表示
        binary_index = bin(i)[2:].zfill(10)  # 转换为10位的二进制表示
        distance = sum(ch1 != ch2 for ch1, ch2 in zip(binary_num, binary_index))
        hamming_distances.append(distance)
    return hamming_distances


def calculate_acc_rate(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("输入的列表长度不一致")

    diff_count = 0
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            diff_count += 1

    difference_rate = diff_count / len(list1)
    return 1 - difference_rate


def calculate_difference_list(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("输入的列表长度不一致")

    difference_list = []
    for i in range(len(list1)):
        difference = list2[i] - list1[i]
        difference_list.append(difference)

    return difference_list


def insert_bit_sorted_new_method(noise, bits):
    # 将张量转换为形状为 [1, 16384] 的张量
    noiselist = noise.view(1, 4 * 64 * 64).clone()

    # 计算绝对值大小并获取排序后的索引
    sorted_indices = torch.argsort(torch.abs(noiselist), dim=1, descending=True)

    sorted_indices_list = sorted_indices.tolist()[0]
    # sorted_indices_2560_2 = sorted_indices_list[:2560 * 2]
    sorted_indices_2560_2 = sorted_indices_list[1024 * 2:1024 * 2 + 2560 * 2]
    shuffled_index_list = list(range(len(sorted_indices_2560_2)))  # 创建保存索引的列表

    random.shuffle(shuffled_index_list)  # 打乱索引列表

    shuffled_indices_2560_2 = [sorted_indices_2560_2[i] for i in shuffled_index_list]  # 使用打乱后的索引列表获取打乱后的元素顺序

    mask_lists = []
    # 2560组，每组2元素 [a, b]
    for i in range(2560):
        # mask_list_i = shuffled_indices_2560_2[i * 2:i * 2 + 2]
        # mask_list_i = sorted_indices_2560_2[i * 2:i * 2 + 2]
        mask_list_i = [sorted_indices_2560_2[i], sorted_indices_2560_2[-(i + 1)]]
        mask_lists.append(mask_list_i)

    # 2560组，每组2元素 [a, b]，按规则藏每个bite
    for i, mask_list in enumerate(mask_lists):
        # 如果嵌入1，a = |a|, b = -|b|
        if bits[i] == 1:
            noiselist[0][mask_list[0]] = abs(noiselist[0][mask_list[0]])
            noiselist[0][mask_list[1]] = -abs(noiselist[0][mask_list[1]])
        # 如果嵌入0，a = -|a|, b = |b|
        else:
            noiselist[0][mask_list[0]] = -abs(noiselist[0][mask_list[0]])
            noiselist[0][mask_list[1]] = abs(noiselist[0][mask_list[1]])

    return noiselist.view(1, 4, 64, 64).clone(), shuffled_index_list, sorted_indices_2560_2


def extract_bit_sorted_new_method(noise_recon, noise_init, shuffled_index_list):
    noise_list_init = noise_init.view(1, 4 * 64 * 64).clone()
    noise_list_recon = noise_recon.view(1, 4 * 64 * 64).clone()
    total_bits = []

    # 计算绝对值大小并获取排序后的索引
    sorted_indices = torch.argsort(torch.abs(noise_list_init), dim=1, descending=True)

    sorted_indices_list = sorted_indices.tolist()[0]
    # sorted_indices_2560_2 = sorted_indices_list[:2560 * 2]
    sorted_indices_2560_2 = sorted_indices_list[1024 * 2:1024 * 2 + 2560 * 2]

    shuffled_indices_2560_2 = [sorted_indices_2560_2[i] for i in shuffled_index_list]  # 使用打乱后的索引列表获取打乱后的元素顺序

    mask_lists = []
    for i in range(2560):
        # mask_list_i = shuffled_indices_2560_2[i * 2:i * 2 + 2]
        # mask_list_i = sorted_indices_2560_2[i * 2:i * 2 + 2]
        mask_list_i = [sorted_indices_2560_2[i], sorted_indices_2560_2[-(i + 1)]]
        mask_lists.append(mask_list_i)

    for i, mask_list in enumerate(mask_lists):
        # 若a>b，bit=1，否则bit=0
        if noise_list_recon[0][mask_list[0]] > noise_list_recon[0][mask_list[1]]:
            total_bits.append(1)
        else:
            total_bits.append(0)

    return total_bits


def insert_bit_sorted_new_method_1024(noise, bits, insert_time):
    # insert_time: 插入到第几个1024*2的位置，插入三次参数分别为为0，1，2
    # 将张量转换为形状为 [1, 16384] 的张量
    noiselist = noise.view(1, 4 * 64 * 64).clone()

    # 计算绝对值大小并获取排序后的索引
    sorted_indices = torch.argsort(torch.abs(noiselist), dim=1, descending=True)

    sorted_indices_list = sorted_indices.tolist()[0]
    # sorted_indices_2560_2 = sorted_indices_list[2560 * 2 + 1024 * 2 * insert_time:2560 * 2 + 1024 * 2 * insert_time + 1024 * 2]
    # sorted_indices_2560_2 = sorted_indices_list[2560 * 2:2560 * 2 + 1024 * 2]
    sorted_indices_2560_2 = sorted_indices_list[:1024 * 2]

    mask_lists = []
    # 2560组，每组2元素 [a, b]
    for i in range(1024):
        mask_list_i = sorted_indices_2560_2[i * 2:i * 2 + 2]
        mask_lists.append(mask_list_i)

    # 1024组，每组2元素 [a, b]，按规则藏每个bite
    for i, mask_list in enumerate(mask_lists):
        # 如果嵌入1，a = |a|, b = -|b|
        if bits[i] == 1:
            noiselist[0][mask_list[0]] = abs(noiselist[0][mask_list[0]])
            noiselist[0][mask_list[1]] = -abs(noiselist[0][mask_list[1]])
        # 如果嵌入0，a = -|a|, b = |b|
        else:
            noiselist[0][mask_list[0]] = -abs(noiselist[0][mask_list[0]])
            noiselist[0][mask_list[1]] = abs(noiselist[0][mask_list[1]])

    return noiselist.view(1, 4, 64, 64).clone()


def extract_bit_sorted_new_method_1024(noise_recon, noise_init, insert_time):
    noise_list_init = noise_init.view(1, 4 * 64 * 64).clone()
    noise_list_recon = noise_recon.view(1, 4 * 64 * 64).clone()
    total_bits = []

    # 计算绝对值大小并获取排序后的索引
    sorted_indices = torch.argsort(torch.abs(noise_list_init), dim=1, descending=True)

    sorted_indices_list = sorted_indices.tolist()[0]
    # sorted_indices_2560_2 = sorted_indices_list[2560 * 2 + 1024 * 2 * insert_time:2560 * 2 + 1024 * 2 * insert_time + 1024 * 2]
    # sorted_indices_2560_2 = sorted_indices_list[2560 * 2:2560 * 2 + 1024 * 2]
    sorted_indices_2560_2 = sorted_indices_list[:1024 * 2]

    mask_lists = []
    for i in range(1024):
        # mask_list_i = shuffled_indices_2560_2[i * 2:i * 2 + 2]
        mask_list_i = sorted_indices_2560_2[i * 2:i * 2 + 2]
        mask_lists.append(mask_list_i)

    for i, mask_list in enumerate(mask_lists):
        # 若a>b，bit=1，否则bit=0
        if noise_list_recon[0][mask_list[0]] > noise_list_recon[0][mask_list[1]]:
            total_bits.append(1)
        else:
            total_bits.append(0)

    return total_bits


# b表示b进制
def convert_binary_to_decimal(binary_list, b):
    decimal_list = []
    for i in range(0, len(binary_list), b):
        binary_slice = binary_list[i:i + b]
        binary_string = ''.join(str(bit) for bit in binary_slice)
        decimal_number = int(binary_string, 2)
        decimal_list.append(decimal_number)
    return decimal_list

# b表示b进制
def convert_decimal_to_binary(decimal_list, b):
    binary_list = []
    for number in decimal_list:
        binary_string = bin(number)[2:].zfill(b)
        binary_slice = [int(bit) for bit in binary_string]
        binary_list.extend(binary_slice)
    return binary_list
