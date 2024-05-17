# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import os
from PIL import Image
from sam_test import entire_img









#sem_path:语义标签路径
#img_path:图片路径
original_sem = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
height, width = original_sem.shape
flatten_sem = original_sem.reshape(-1)

masks = entire_img(img_path)

for i in range(len(masks)):
        mask = masks[i]['segmentation'].reshape(-1)
        now_sem = flatten_sem[mask]
        label_counts = np.bincount(now_sem, minlength=19)
        most_commom_label = np.argmax(label_counts)
        most_commom_count = label_counts[most_commom_label]

        # 如果该标签过半数,那么该mask内都附上这个标签
        if most_commom_count > (len(now_sem) // 2):
            print((flatten_sem[mask]!=most_commom_label).sum())
            flatten_sem[mask] = most_commom_label
            print((flatten_sem[mask]!=most_commom_label).sum())

refine_sem = flatten_sem.reshape(height, width)
cv2.imwrite(save_sem_path, np.uint8(refine_sem))