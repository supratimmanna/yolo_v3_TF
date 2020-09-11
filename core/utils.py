import os
import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from core.config import cfg

### To read all the class names
def read_class_names(class_file_name):
    names={}
    with open(class_file_name,'r') as f:
        for ID, name in enumerate(f):
            names[ID] = name.strip('\n')
    return names

#### To read all the anchors
def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3,3,2)

### Preprocess the input images

def preprocessing_input_image(image, target_size, gt_bbox=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    target_h, target_w = target_size
    h, w, _ = image.shap 
    
    scale = min(target_h/h, target_w/w)
    
    new_h, new_w = (scale*h, scale*w)
    d_h, d_w = (target_h-new_h)//2, (target_w-new_w)//2
    image_resize = cv2.resize(image,(new_h, new_w))
    image_padded  = np.full(shape=[target_h, target_w, 3], fill_value=128.0)
    image_padded[d_h:d_h+new_h, d_w:d_w+new_w,:] = image_resize
    image_padded = image_padded/255.0
    
    if gt_bbox is None:
        return image_padded
    
    else:
        gt_bbox[:,[0, 2]] = gt_bbox[:,[0, 2]]*scale + d_w
        gt_bbox[:,[1, 3]] = gt_bbox[:,[1, 3]]*scale + d_h
        
    return image_padded, gt_bbox