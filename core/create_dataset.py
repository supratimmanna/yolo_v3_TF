import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg

class Create_Dataset(object):
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_size = cfg.TRAIN.INPUT_SIZE if dataset_type =='train' else cfg.TEST.INPUT_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type =='train' else cfg.TEST.BATCH_SIZE
        self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type =='train' else cfg.TEST.DATA_AUG
        
        
        self.train_input_size = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchors_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150
        
        self.annotations = self.load_annotation(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batches = int(np.ceil(self.num_samples/self.batch_size))
        self.batch_count = 0
        
        
    def load_annotation(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations
    
    
    def __iter__(self):
        return self
    
    def __next__(self):
        with tf.device('\cpu:0'):
            self.train_input_size = random.choice(self.train_input_size)
            self.train_output_size = self.train_input_size//self.strides
            
            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))
            
            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_size[0], self.train_output_size[0],
                                          self.anchors_per_scale, 5+self.num_classes))
            
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_size[1], self.train_output_size[1],
                                          self.anchors_per_scale, 5+self.num_classes))
            
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_size[1], self.train_output_size[1],
                                          self.anchors_per_scale, 5+self.num_classes))
            
            batch_sbbox = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbbox = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbbox = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            
            num=0
            
            if self.batch_count < self.num_batches:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index = - self.num_samples
                        
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotaion(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.true_boxes_preprocessing(bboxes)
                    
                    
                    
    def parse_annotaion(self, annotation):
        
        details = annotation.split()
        image_path = details[0]
        if not os.path.exists(image_path):
            raise KeyError("%s doesn't exist" %image_path)
        
        image = np.array(cv2.imread(image_path))
        bbox=[]
        for box in details[1:]:
            a=box.split(',')
            bbox.append(a)
        bboxes=np.array(bbox,dtype='int32')
        
        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))
            
        image, bboxes = utils.preprocessing_input_image(np.copy(image), [self.train_input_size, self.train_input_size],
                                                        np.copy(bboxes))
        
        return image, bboxes
            
            
            
            
    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes
    
    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes
    
    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes
    
    def iou(self, bbox1, bbox2):
        ### bbox : (x,y,w,h) so area = w*h
        
        bbox1_area = bbox1[2] * bbox1[3]
        bbox2_area = bbox2[2] * bbox2[3]
        
        bbox1_cords = tf.concat([bbox1[:2] - bbox1[2:] * 0.5, bbox1[:2] + bbox1[2:] * 0.5], axis=-1)
        bbox2_cords = tf.concat([bbox2[:2] - bbox2[2:] * 0.5, bbox2[:2] + bbox2[2:] * 0.5], axis=-1)
        
        left_up = tf.maximum(bbox1_cords[:2], bbox2_cords[:2])
        right_bottom = tf.minimum(bbox2_cords[2:], bbox2_cords[2:])
        
        intersection = tf.maximum((right_bottom - left_up), 0.0)
        
        intersection_area = intersection[0] * intersection[1]
        
        union_area = bbox1_area + bbox2_area -intersection_area
        
        iou = (intersection_area / (union_area + 1e-10)) ### 1e-10 is used to avoid zero in the denominator
        
        return iou, union_area
    
    def true_boxes_preprocessing(self, bboxes):
         label = [np.zeros((self.train_output_size[i], self.train_output_size[i], self.anchors_per_scale,
                            5+self.num_classes)) for i in range(3)]
            
         bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range (3)]
         
         bbox_count = np.zeros((3,))
         
         for bbox in bboxes:
             bbox_coord = bbox[:4]
             bbox_class_ind = bbox[4]
             
             onehot = np.zeros(self.num_classes, dtype=np.float)
             onehot[bbox_class_ind] = 1
             uniform_distribution = np.full(self.num_classes, 1.0/self.num_classes)
             delta = 0.01
             smooth_onehot = (1.0-delta)*onehot + delta*uniform_distribution
             
             bbox_xywh = np.concatenate((bbox_coord[:2]+bbox_coord[2:])*0.5, (bbox_coord[2:]-bbox_coord[:2]), axis=-1)
             bbox_xywh_scaled = bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
             
             iou=[]
             exist_positive = False
             
             for i in range(3):
                 anchors_xywh = np.zeros(self.anchors_per_scale, 4)
                 anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                 anchors_xywh[:,2:] = self.anchors[i]
                 
                 iou_value = self.iou(bbox_xywh_scaled[i][np.newaxis,:], anchors_xywh)
                 iou.append(iou_value)
                 iou_mask = iou_value > 0.3
                 
                 if np.any(iou_mask):
                     
                     xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                     
                     label[i][yind, xind, iou_mask, :] = 0
                     label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                     label[i][yind, xind, iou_mask, 4:5] = 1.0
                     label[i][yind, xind, iou_mask, 5:] = smooth_onehot
                     
                     bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                     bboxes_xywh[i][bbox_ind, 0:4] = bbox_xywh
                     bbox_count[i] = +1
                     exist_positive = True
             if not exist_positive:
                 best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                 best_detect = int(best_anchor_ind / self.anchor_per_scale)
                 best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                 xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                 label[best_detect][yind, xind, best_anchor, :] = 0
                 label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                 label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                 label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                 bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                 bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                 bbox_count[best_detect] += 1                     
                
         label_sbbox, label_mbbox, label_lbbox = label
         sbboxes, mbboxes, lbboxes = bboxes_xywh
        
         return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
    
    
    def __len__(self):
        return self.num_batches

             
         
         
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            