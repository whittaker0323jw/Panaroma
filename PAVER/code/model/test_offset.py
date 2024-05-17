"""Vision Transformer

PyTorch implementation adapted from timm (20b2d4b) library by rwrightman
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

"""

import math
# from functools import partial
# from collections import OrderedDict

import torch
# import panda as pd
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.ops import DeformConv2d

# from exp import ex
# from utils import download_url
# from . import full_model, logger
# from .vit_utils import to_2tuple, DropPath, lecun_normal_, trunc_normal_, \
#                        resize_pos_embed, adapt_input_conv, resize_pos_embed
import numpy as np

from geometry_test import compute_deform_offset



offset = torch.from_numpy(compute_deform_offset(model_config= None,
                                                        is_discrete= False
                                                        )).float()
# reshaped_offset = offset.reshape(7*7, 2, 100, 512)
# reshaped_offset = offset.reshape(16*16, 2, 14, 28)
# reshaped_offset = offset.reshape(16, 2, 256, 512)
# print(offset.reshape(49,2,256,512)[:,0,128,0]) #torch.Size([32, 256, 512])32,56,112 98,32,64
tensor = offset.reshape(98,128,256)
torch.save(tensor, 'offset_tensor_512x1024.pt')
# start_idx = (tensor.size(1) - 100) // 2
# end_idx = start_idx + 100
# selected_tensor = tensor[:, start_idx:end_idx, :]
# tensor = torch.unsqueeze(selected_tensor, 0)

# 保存张量到文件

#copied_tensor = tensor.repeat(4, 1, 1, 1)
# print(offset.shape)
# data = reshaped_offset[:, 1, 0,0].reshape(16, 16)#顺序：y，x
# print(data)
# offset_x = reshaped_offset[:,:,0,:,:].reshape(16,16,14,28)#.transpose((2, 0, 3, 1)).reshape(16*14, 16*28)
# offset_y = reshaped_offset[:,:,1,:,:].reshape(16,16,14,28)#.transpose((2, 0, 3, 1)).reshape(16*14, 16*28)
# offset_x = offset_x.permute(2, 0, 3, 1).reshape(16*14, 16*28)#transpose一次只能调换两个参数*224*448
# np.savetxt('offset_x.txt', offset_x)
#维度转换错了。。。。。。。。
# print(reshaped_offset[:,0,13,27])
# print("saved")
# data = reshaped_offset[:, 1, 0,0].reshape(16, 16)#顺序：y，x
# print(data)

# 创建一个 DataFrame 对象
# df = pd.DataFrame(data)

# 将 DataFrame 写入 Excel 文件
# df.to_excel('output.xlsx', index=False)

# print("Data saved to output.xlsx")