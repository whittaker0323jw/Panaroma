import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from model.trans4pass import Trans4PASS_v1, Trans4PASS_v2
from model.trans4passplus import Trans4PASS_plus_v1,Trans4PASS_plus_v2
from dataset.densepass_dataset import densepassDataSet, densepassTestDataSet
from collections import OrderedDict
from SAM.sam_test import entire_img
import os
from PIL import Image
import cv2

import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


TARGET_NAME = 'DensePASS'
DATA_DIRECTORY = './datasets/DensePASS'
DATA_LIST_PATH = './dataset/densepass_list/train.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 19
BATCH_SIZE = 1
NUM_WORKERS = 0
MODEL = 'Trans4PASS_plus_v1'
# RESTORE_FROM = 'snapshots/CS2DensePASS_Trans4PASS_v1_WarmUp/BestCS2DensePASS_G.pth'
# RESTORE_FROM = 'workdirs/cityscapes/trans4pass_plus_tiny_512x512/trans4pass_plus_tiny_512x512.pth' #调用的是总的模型（pth full）
# RESTORE_FROM = '/home/w/Downloads/Trans4PASS-main/adaptations/snapshots/CS2DP_Trans4PASS_plus_v1_MPA/BestCS2DensePASS_G.pth'
RESTORE_FROM = 'snapshots/CS2DensePASS_Trans4PASS_plus_v1_WarmUp/BestCS2DensePASS_14000iter_50.01miou.pth'
SET = 'train'
SAVE_PATH = './pseudo_SAM_test_{}_{}_ms'.format(TARGET_NAME, MODEL)
EMB_CHANS = 128

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice Deeplab.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'Trans4PASS_v1':
        model = Trans4PASS_v1(num_classes=args.num_classes)
    elif args.model == 'Trans4PASS_v2':
        model = Trans4PASS_v2(num_classes=args.num_classes)
    elif args.model == 'Trans4PASS_plus_v1':
        model = Trans4PASS_plus_v1(num_classes=args.num_classes, emb_chans=EMB_CHANS)
    elif args.model == 'Trans4PASS_plus_v2':
            model = Trans4PASS_plus_v2(num_classes=args.num_classes, emb_chans=EMB_CHANS)
    else:
        raise ValueError
    saved_state_dict = torch.load(RESTORE_FROM, map_location=lambda storage, loc: storage)
    if 'state_dict' in saved_state_dict.keys():
        saved_state_dict = saved_state_dict['state_dict']
    msg = model.load_state_dict(saved_state_dict, strict=False)
    print(msg)

    device = torch.device("cuda" if not args.cpu else "cpu")
    model = model.to(device)
    model.eval()
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    targetset = densepassDataSet(args.data_dir, args.data_list, crop_size=(2048,400), set=args.set)
    testloader = data.DataLoader(targetset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    interp = nn.Upsample(size=(400, 2048), mode='bilinear', align_corners=True)
    predicted_label = np.zeros((len(targetset), 400, 2048), dtype=np.int8)
    predicted_prob = np.zeros((len(targetset), 400, 2048), dtype=np.float16)
    image_name = []
    count = 0
    for index, batch in enumerate(testloader):
        if index % 10 == 0:
            print('{}/{} processed'.format(index, len(testloader)))

        image, _, name = batch
        image_name.append(name[0])
        image = image.to(device)
        # images.append(image)#####################
        b, c, h, w = image.shape
        output_temp = torch.zeros((b, 19, h, w), dtype=image.dtype).to(device)
        scales = [0.5,0.75,1.0,1.25,1.5,1.75]
        for sc in scales:
            new_h, new_w = int(sc * h), int(sc * w)
            img_tem = nn.UpsamplingBilinear2d(size=(new_h, new_w))(image)
            with torch.no_grad():
                _, output = model(img_tem)
                output_temp += interp(output) #resize到原尺寸
        output = output_temp / len(scales)
        output = F.softmax(output, dim=1)
        output = interp(output).cpu().data[0].numpy()#转到cpu上可能有一些问题？？
        output = output.transpose(1,2,0)
        
        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        predicted_label[index] = label#测值最大的索引
        predicted_prob[index] = prob#预测的概率

        #少数量样本测试部分
        count = count +1
        # if(count>50):
        #     break
        #少数量样本测试部分结束
        thres = [0.9,0.865,  0.9,    0.786,  0.7676, 0.725,  0.7275, 0.727,  0.9 ,   0.9 ,0.9  ,  0.7563, 0.6045 ,0.9   , 0.7734 ,0.9   , 0.9   , 0.7964 ,0.7617]


# #######################插入部分####################################      
        for i in range(1,19):
            label[(prob<thres[i])*(label==i)] = 255  
        output = np.asarray(label, dtype=np.uint8)


        #######################################3
        ##应用SAM进行校正#########################
        
        height, width = output.shape
        flatten_sem = output.reshape(-1)#原来的结果 #
        # print(flatten_sem.shape)#819200
        

        # print(image.shape)#<class 'tuple'><class 'torch.Tensor'>torch.Size([3, 400, 2048])
        image_np = image[0].cpu().permute(1,2,0).numpy().astype('uint8')
        # print(image_np.size())
        # print("image_shape:", image_np.shape)
        masks = entire_img(image_np)
        
        for i in range(len(masks)):
                # print("masks_shape:",masks[i]['segmentation'].shape)
                mask = masks[i]['segmentation'].reshape(-1)
                now_sem = flatten_sem[mask]
                label_counts = np.bincount(now_sem, minlength=19)
                most_commom_label = np.argmax(label_counts)
                most_commom_count = label_counts[most_commom_label]
            
                # 如果该标签过半数,那么该mask内都附上这个标签
                if most_commom_count > (len(now_sem) // 2):
                    # print((flatten_sem[mask]!=most_commom_label).sum())#这个语句计算了 flatten_sem 数组中在 mask 为 True 的位置上，与 most_common_label 不同的元素数量。
                    print("true")
                    flatten_sem[mask] = most_commom_label
                    # print((flatten_sem[mask]!=most_commom_label).sum())
                else:
                    print("false")

        refine_sem = flatten_sem.reshape(height, width)#type?
        output = np.asarray(refine_sem,dtype=np.uint8)
        output_col = colorize_mask(output)
        print(output.size)
        print(output_col.size)
        ##应用SAM进行校正结束#########################
        #######################################3
        
        
        output = Image.fromarray(output)
        name = name[0].replace('.png', '_labelTrainIds.png')
        save_fn = os.path.join(args.save, name)
        if not os.path.exists(os.path.dirname(save_fn)):
            os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        # args.save = './pseudo_SAM_test_{}_{}_ms_color'.format(TARGET_NAME, MODEL)
        # output_col.save('%s/%s_color.png' % (args.save, name))
        output.save(save_fn)

###########################end##################################3    
    
    thres = []
    thres_small = []
    for i in range(19):
        x = predicted_prob[predicted_label==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        if(i == 0):
            print("x:",x)
            print("lenx:",len(x));
            print("0:",np.count_nonzero(x == 0))
        thres.append(x[int(np.round(len(x)*0.5))])#计算排序后的预测概率数组的中位数，作为这个类别的阈值。
        thres_small.append(x[int(np.round(len(x)*0.3))])

       #
    thres = np.array(thres)
    thres_small = np.array(thres_small)
    thres[thres>0.9]=0.9
    thres_small[thres_small>0.9]=0.9
    print(thres)
    print(thres_small)
    # exit()
    # for index in range(len(targetset)//BATCH_SIZE):
    for index in range(len(image_name)//BATCH_SIZE):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(1,19):
            label[(prob<thres[i])*(label==i)] = 255  
        output = np.asarray(label, dtype=np.uint8)


        #######################################3
        ##应用SAM进行校正#########################
        
        # height, width = output.shape
        # flatten_sem = output.reshape(-1)#原来的结果 #
        # # print(flatten_sem.shape)#819200
        # image = testloader.dataset[index][0]

        # #print(image.shape)#<class 'tuple'><class 'torch.Tensor'>torch.Size([3, 400, 2048])
        # image_np = image.permute(1,2,0).numpy().astype('uint8')
        # # print(image_np.size())
        # # print("image_shape:", image_np.shape)
        # masks = entire_img(image_np)
        
        # for i in range(len(masks)):
        #         # print("masks_shape:",masks[i]['segmentation'].shape)
        #         mask = masks[i]['segmentation'].reshape(-1)
        #         now_sem = flatten_sem[mask]
        #         label_counts = np.bincount(now_sem, minlength=19)
        #         most_commom_label = np.argmax(label_counts)
        #         most_commom_count = label_counts[most_commom_label]
            
        #         # 如果该标签过半数,那么该mask内都附上这个标签
        #         if most_commom_count > (len(now_sem) // 2):
        #             # print((flatten_sem[mask]!=most_commom_label).sum())#这个语句计算了 flatten_sem 数组中在 mask 为 True 的位置上，与 most_common_label 不同的元素数量。
        #             flatten_sem[mask] = most_commom_label
        #             # print((flatten_sem[mask]!=most_commom_label).sum())
        #         else:
        #             print("false")

        # refine_sem = flatten_sem.reshape(height, width)#type?
        # output = np.asarray(refine_sem,dtype=np.uint8)
        output_col = colorize_mask(output)
        # print(output.size)
        # print(output_col.size)
        ##应用SAM进行校正结束#########################
        #######################################3
        
        
        output = Image.fromarray(output)
        name = name.replace('.png', '_labelTrainIds.png')
        save_fn = os.path.join(args.save, name)
        if not os.path.exists(os.path.dirname(save_fn)):
            os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        # args.save = './pseudo_SAM_test_{}_{}_ms_color'.format(TARGET_NAME, MODEL)
        # output_col.save('%s/%s_color.png' % (args.save, name))
        output.save(save_fn)

if __name__ == '__main__':
    main()
