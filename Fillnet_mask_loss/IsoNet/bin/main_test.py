# -*- coding: utf-8 -*-
import numpy as np
class one():
    def m(self,string:str='dsa'):
        print(string)
def get_turn_notation_list(notation): #一共获得24种不同方向图像
    turn_notation_list = []
    for i in range(4):
        temp_notation = np.rot90(notation,k = i,axes=(1, 2)) #向下翻转
        for a in range(4):
            last_notation = np.rot90(temp_notation, k = a,axes=(2, 3)) #逆时针翻转
            turn_notation_list.append(last_notation)

    temp_notation = np.rot90(notation, k=1, axes=(1, 3))#向右翻转
    for a in range(4):
        last_notation = np.rot90(temp_notation, k=a, axes=(2, 3))  # 逆时针翻转
        turn_notation_list.append(last_notation)

    temp_notation = np.rot90(notation, k=1, axes=(3, 1))  # 向左翻转
    for a in range(4):
        last_notation = np.rot90(temp_notation, k=a, axes=(2, 3))  # 逆时针翻转
        turn_notation_list.append(last_notation)
    return turn_notation_list
def recover_turn_notation_list(turn_notation_list): # 将 这24种不同方向的图像复原
    recover_notation_list = []
    for i in range(4):#先将向下翻转的矩阵翻转
        for a in range(4):
            temp_notation = np.rot90(turn_notation_list[i * 4 + a], k = a,axes=(2, 1))
            temp_notation = np.rot90(temp_notation, k = i,axes=(1, 0))
            recover_notation_list.append(temp_notation)

    for i in range(4):#把向右翻转的矩阵翻转
        temp_notation = np.rot90(turn_notation_list[16+i], k = i, axes=(2, 1))
        temp_notation = np.rot90(temp_notation, k=1, axes=(2, 0))
        recover_notation_list.append(temp_notation)

    for i in range(4):#把向左翻转的矩阵翻转
        temp_notation = np.rot90(turn_notation_list[20+i], k = i, axes=(2, 1))
        temp_notation = np.rot90(temp_notation, k=1, axes=(0, 2))
        recover_notation_list.append(temp_notation)
    return recover_notation_list
if __name__ == '__main__':
    # import torch
    # import torch.nn as nn
    # from IsoNet.models.SwinUnet.networks.swin_transformer_unet_skip_expand_decoder_sys import PatchEmbed
    # # 定义输入数据
    # input_data = torch.randn(1, 64, 64, 64)  # 1个样本，3个通道，64x64尺寸
    # # 初始化PatchEmbed模块
    # patch_embed = PatchEmbed(img_size=64, patch_size=4, in_chans=64, embed_dim=96)
    #
    # # 将输入数据传递给PatchEmbed模块
    # output = patch_embed(input_data)
    #
    # print("Output shape:", output.shape)  # 打印输出形状

    # out = [1,2,3,4,5,6,7,8,9,10]
    # out  = out>np.percentile(out, 100-70)
    # print(out)
    # isonet = isonet.ISONET()
    # isonet.deconv(star_file='E:\\workspace\\py_workspace\\py_fold\\IsoNet-master\\data_isonet\\HIV\\demo_data\\tomograms\\TS01-wbp.rec')
    # isonet.extract(
    #   star_file='E:\\workspace\\py_workspace\\py_fold\\IsoNet-master\\data_isonet\\HIV\\demo_data\\tomograms\\TS01-wbp.rec')

    # a =  np.array(
    #     [
    #     [[1,2],
    #      [3,4]],
    #
    #     [[5,6],
    #      [7,8]]
    #     ])
    # rotated_array = np.rot90(a, k=4,axes=(0, 1))
    # print(rotated_array)
    # 定义3D数组
    a = np.array([[[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]]])
    a_list = get_turn_notation_list(a)
    print(a_list)
    a_list = np.squeeze(a_list)
    recover_list = recover_turn_notation_list(a_list)
    print(recover_list)
    # # 打印原始数组
    # print("Original Array:")
    # print(a)
    # rotated_a = np.rot90(a, k=1, axes=(1, 2))#向下翻转
    # print("*"*100)
    # print(rotated_a)
    # rotated_a = np.rot90(a, k=1, axes=(2, 3))#逆时针翻转
    # print("*"*100)
    # print(rotated_a)
    # rotated_a = np.rot90(a, k=1, axes=(1, 3))#向右翻转
    # print("*"*100)
    # print(rotated_a)