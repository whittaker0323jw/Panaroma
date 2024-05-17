import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from packaging import version

import os, sys
os.chdir(sys.path[0])
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.trans4pass import Trans4PASS_v1, Trans4PASS_v2
from dataset.densepass_dataset import densepassDataSet, densepassTestDataSet
from dataset.PANO_dataset import PANODataSet, PANOTestDataSet
from dataset.cs_dataset_src import CSSrcDataSet
from torchvision import transforms
from compute_iou import fast_hist, per_class_iu

from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
torch.cuda.empty_cache()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
SOURCE_NAME = 'PANO'
TARGET_NAME = 'DensePASS'
MODEL = 'Trans4PASS_v1'
DIR_NAME = '{}2{}_{}_MPA/'.format(SOURCE_NAME, TARGET_NAME, MODEL)
DATA_DIRECTORY = 'datasets/PANO'
DATA_LIST_PATH = 'dataset/PANO_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
SAVE_PATH = './result/' + DIR_NAME

IGNORE_LABEL = 255
NUM_CLASSES = 19
RESTORE_FROM = 'snapshots/CS2DensePASS_Trans4PASS_v1_MPA/BestCS2DensePASS_G.pth'
SET = 'val'

EMB_CHANS = 128
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)



NAME_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "light",
    "sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motocycle",
    "bicycle"]

for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')# 将 Numpy 数组转换为 PIL 图像对象
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()
#################self-added#######################################3
def png_to_tensor(image_path):
    # 打开图片
    img = Image.open(image_path)

    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 调整大小到512x512
        transforms.ToTensor()  # 将 PIL 图像转换为张量
    ])

    # 应用转换
    tensor_img = transform(img)

    return tensor_img
def mix(mask, pin, pan):
    data = torch.cat(mask * pin + (1 - mask) * pan)
    return data
                      
#################self-added#######################################3
#################self-added#######################################3
def main():
    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'Trans4PASS_v1':
        model = Trans4PASS_v1(num_classes=args.num_classes, emb_chans=EMB_CHANS)
    elif args.model == 'Trans4PASS_v2':
        model = Trans4PASS_v2(num_classes=args.num_classes, emb_chans=EMB_CHANS)
    else:
        raise ValueError

    saved_state_dict = torch.load(args.restore_from, map_location='cuda:0')
    if 'state_dict' in saved_state_dict.keys():
        saved_state_dict = saved_state_dict['state_dict']
    msg = model.load_state_dict(saved_state_dict, strict=False)
    print(msg)

    model.eval()
    model.cuda(gpu0)

    w, h = map(int, args.input_size_target.split(','))
    targettestset = PANOTestDataSet(args.data_dir, args.data_list, crop_size=(w, h),
                                         mean=IMG_MEAN,
                                         scale=False, mirror=False, set='train')
    testloader = data.DataLoader(targettestset, batch_size=1, shuffle=False, pin_memory=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    hist = np.zeros((args.num_classes, args.num_classes))
    interp = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)



    ##################self-added#########################################################  
    img_pin = png_to_tensor('projection_test/city_512x512.png').unsqueeze(0).cuda(gpu0)
    img_pan = png_to_tensor('projection_test/den_512x512.png').unsqueeze(0).cuda(gpu0)
    with torch.no_grad():
                #print(type(img_tem))#<class 'torch.Tensor'>
            _, output_pin = model(img_pin)
            _, output_pan = model(img_pan)
            #print(output_pan.shape)#torch.Size([1, 19, 512, 512])
    output_pin = torch.argmax(output_pin, 1).squeeze(0).cpu().data
    output_pan = torch.argmax(output_pan, 1).squeeze(0).cpu().data
    # print(output_pin.shape)#512x512
    ###mask
    classes = torch.unique(output_pan)
    
    nclasses = classes.shape[0]
    #halfed—classes
    classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()])
    # print(classes)#9
    mask = torch.tensor([1 if val in classes else 0 for val in output_pan.view(-1)]).view(output_pan.size())
    expanded_mask = mask.unsqueeze(0).expand_as(img_pin.squeeze(0))
    # print(expanded_mask.shape)#3x512x512
    combined_tensor = img_pan.squeeze(0).cpu() * expanded_mask + img_pin.squeeze(0).cpu()* (1 - expanded_mask)
    # print(combined_tensor.shape)#3x512x512
    image_tensor = combined_tensor.clone().detach()
        # 将张量转换为图像
    image_tensor = transforms.ToPILImage()(image_tensor)
        # 保存图像
    image_tensor.save(os.path.join('projection_test', "combined_0.png"))
    print("saved@.@!!")


    mix_color = output_pin* (1 - mask)+ output_pan* mask
    print(mix_color.shape)#torch.Size([3, 512, 512])
    output_col = colorize_mask(mix_color.numpy())
    
    output_col.save(os.path.join('projection_test', "colored_combined_0.png"))
    # print(mask.shape)#512x512
    # mixed_tensor = mix(mask,img_pin.squeeze(0),img_pan.squeeze(0))
    ##################self-added######################################################### 





    # img_city = cv2.read()

    # mIoUs = per_class_iu(hist)
    # for ind_class in range(args.num_classes):
    #     print('===>{:<15}:\t{}'.format(NAME_CLASSES[ind_class], str(round(mIoUs[ind_class] * 100, 2))))
    # bestIoU = round(np.nanmean(mIoUs) * 100, 2)
    # print('===> mIoU: ' + str(bestIoU))


if __name__ == '__main__':
    main()
