# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from SAM.segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import os
from PIL import Image


def process_img(image):
    '''img_path to img(np.array)
    '''
    # image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 获取原始图像的尺寸
    # original_height, original_width = image.shape[:2]
    
    # 计算降采样后的尺寸
    # new_height = original_height // 4
    # new_width = original_width // 4
    
    # 使用双线性插值对图像进行降采样
    # resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return image

def entire_img(img_path, 
               model_type = "vit_b", model_path="/home/w/Downloads/Trans4PASS-main/adaptations/SAM/sam_vit_b_01ec64.pth"):
    '''whole img generate mask
    '''
    #输入numpy数组
    # image = process_img(img_path)
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device="cpu")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img_path)
    return masks

def predict(img_path, 
            input_point=None, input_label=None,
            input_box=None,
            type='point', 
            multimask_output=True,
            model_type="vit_b", model_path="sam_vit_b_01ec64.pth"):
    image = process_img(img_path)
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device="cuda")

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    if type == 'point':
        masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
        )
    elif type == 'box':
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
    elif type == 'both':
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :],
            multimask_output=False,
        )
    
    return masks, scores, logits
    
def main():
    # img_path = './images/0000002534.png'
    img_path = './images/00000000000000629.jpg'
    model_type = "vit_b"
    model_path = "sam_vit_b_01ec64.pth"
    output_path = "./output_masks"
    
    # (1) 整个图像统一处理，得到若干个mask
    masks = entire_img(img_path,    
                       model_type=model_type, model_path=model_path) # 一个列表，每个元素都是一个mask(dict对象)
    # 每个mask包含 ['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box']
    # print(masks[0]['segmentation'].shape)
    print(len(masks))#40个类别
    

    # (2) 输入一个或多个点prompt和他们的label，得到这个点相关的masks
    # 注：label指这两个点是不是一类
    # 注：masks会默认返回3个，有可能的三个大小不一的mask；通过multimask_output指定需不需要
    # 一个点
    # input_point = np.array([[1064, 1205]])
    # input_label = np.array([1])
    # 多个点
    input_point = np.array([[1064, 1205], [1111, 1222]])
    input_label = np.array([1, 1])
    masks, scores, logits = predict(img_path, 
                                    type='point',
                                    input_point=input_point, input_label=input_label,
                                    multimask_output=True,
                                    model_type=model_type, model_path=model_path)
    
    # print(masks[0].shape)#(376, 1408)
    # print(len(masks))#3
    # print(masks[0])#True...False...
    # print(masks[1])#False all
    # print(masks[2])#False all
    # mask0 = Image.fromarray(masks[0])
    # mask1 = Image.fromarray(masks[1])
    # mask2 = Image.fromarray(masks[2])
    # file_path0 = os.path.join(output_path, 'mask0.png')
    # file_path1 = os.path.join(output_path, 'mask1.png')
    # file_path2 = os.path.join(output_path, 'mask2.png')
    # mask0.save(file_path0)
    # mask1.save(file_path1)
    # mask2.save(file_path2)

    # (3) 输入一个包围框作为prompt，得到这个框内的mask
    input_box = np.array([1305, 244, 2143, 1466])
    masks, scores, logits = predict(img_path, 
                                    type='box',
                                    input_box=input_box,
                                    multimask_output=True,
                                    model_type=model_type, model_path=model_path)
    print(masks[0].shape)
    sur = Image.fromarray(masks[0])
    file_path0 = os.path.join(output_path, 'sur.jpg')
    sur.save(file_path0)
    # (4) 点和包围框都作为输入
    input_point = np.array([[1064, 1205], [1111, 1222]])
    input_label = np.array([1, 1])
    input_box = np.array([1305, 244, 2143, 1466])
    masks, scores, logits = predict(img_path, 
                                    type='both',
                                    input_point=input_point, input_label=input_label,
                                    input_box=input_box,
                                    multimask_output=False,
                                    model_type=model_type, model_path=model_path)
    print(masks[0].shape)
    
    # 更多阅读https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb

if __name__ == "__main__":
    main()