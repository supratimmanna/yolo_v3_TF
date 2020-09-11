import numpy as np
import tensorflow as tf
import core.common as common
import core.darknet_backbone as backbone
import core.utils as utils
import core.config as cfg

class YOLOv3(object):
    
    def __init_(self, input_data, trainable):
        
        self.trainable = trainable
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class = len(self.classes)
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh = cfg.YOLO.iou_loss_threshself
        self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD
        
        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.object_detection_network(input_data)
        except:
            raise NotImplementedError("YOLOv3 network can't be built")
            
        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])
            
        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])
            
        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])
            
            
    def object_detection_network(self, input_data):
        
        route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)
        
        conv_output = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv53')
        
        conv_output = common.convolutional(conv_output, (3, 3, 512, 1024), self.trainable, 'conv54')
        
        conv_output = common.convolutional(conv_output, (1, 1, 1024, 512), self.trainable, 'conv55')
        
        conv_output = common.convolutional(conv_output, (3, 3, 512, 1024), self.trainable, 'conv56')
        
        conv_output = common.convolutional(conv_output, (1, 1, 1024, 512), self.trainable, 'conv57')
        
        conv_lobj_branch = common.convolutional(conv_output, (3, 3, 512, 1024), self.trainable, 'conv_lobj_branch')
        
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(self.num_class+5)), 
                                          trainable=self.trainable, activate=False, bn=False,
                                          name='conv_lbbox')
        
        conv_output = common.convolutional(conv_output, (1, 1, 512, 256), self.trainable, 'conv58')
        
        conv_upsampled_1 = common.upsample(name='upsample1', input_data=conv_output,
                                           method=self.upsample_method)
        
        with tf.variable_scope('route_1'):
            conv_output = tf.concat([conv_upsampled_1, route_2], axis=-1)
            
            conv_output = common.convolutional(conv_output, (1,1,768,256), self.trainable, 'conv59')
            
            conv_output = common.convolutional(conv_output, (3,3,256,512), self.trainable, 'conv60')
            
            conv_output = common.convolutional(conv_output, (1,1,512,256), self.trainable, 'conv61')
            
            conv_output = common.convolutional(conv_output, (3,3,256,512), self.trainable, 'conv62')
            
            conv_output = common.convolutional(conv_output, (1,1,512,256), self.trainable, 'conv63')
            
            conv_mobj_branch = common.convolutional(conv_output, (3,3,256,512), self.trainable, 'conv_mobj_branch')
            
            conv_mbbox = common.convolutional(conv_mobj_branch, (1,1,512,3*(self.num_class+5)),
                                              self.trainable, activate=False, bn=False,
                                              name ='conv_mbbox' )
            
            conv_output = common.convolutional(conv_output, (1, 1, 256, 128), self.trainable, 'conv64')
        
            conv_upsampled_2 = common.upsample(name='upsample2', input_data=conv_output,
                                           method=self.upsample_method)
            
        with tf.variable_scope('route_2'):
            conv_output = tf.concat([conv_upsampled_2, route_1], axis=-1)
            
            conv_output = common.convolutional(conv_output, (1,1,384,128), self.trainable, 'conv65')
            
            conv_output = common.convolutional(conv_output, (3,3,128,256), self.trainable, 'conv66')
            
            conv_output = common.convolutional(conv_output, (1,1,256,128), self.trainable, 'conv67')
            
            conv_output = common.convolutional(conv_output, (3,3,128,256), self.trainable, 'conv68')
            
            conv_output = common.convolutional(conv_output, (1,1,256,128), self.trainable, 'conv69')
            
            conv_sobj_branch = common.convolutional(conv_output, (3,3,128,256), self.trainable, 'conv70')
            
            conv_sbbox = common.convolutional(conv_sobj_branch, (1,1,256,3*(self.num_class+5)),
                                              self.trainable, activate=False, bn=False,
                                              name = 'conv_sbbox')
            
        return conv_lbbox, conv_mbbox, conv_sbbox
        
    
    def decode_bbox(self, conv_bbox, anchors, stride):
        
        conv_shape = tf.shape(conv_bbox)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)
        
        conv_bbox = tf.reshape(conv_bbox, (batch_size, output_size, output_size, anchor_per_scale, 5+self.num_class))
        
        conv_bbox_raw_xy = conv_bbox[:, :, :, :, 0:2]
        conv_bbox_raw_hw = conv_bbox[:, :, :, :, 2:4]
        conv_bbox_raw_conf = conv_bbox[:, :, :, :, 4:5]
        conv_bbox_raw_prob = conv_bbox[:, :, :, :, 5:]
        
        #x = tf.tile(tf.range(13, dtype=tf.int32)[:, tf.newaxis], [1,13])
        x = tf.range(13, dtype=tf.int32)
        x = tf.expand_dims(x, axis=-1)
        x = tf.tile(x,[1,13])
        
        #y = tf.tile(tf.range(13, dtype=tf.int32)[:, tf.newaxis], [1,13])
        y = tf.range(13, dtype=tf.int32)
        y = tf.expand_dims(y, axis=-1)
        y = tf.tile(y,[1,13])
        
        #xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.concate([tf.expand_dims(x,axis=-1), tf.expand_dims(y,axis=-1)], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)
        
        pred_xy = (tf.sigmoid(conv_bbox_raw_xy) + xy_grid)* stride
        pred_hw = (tf.exp(conv_bbox_raw_hw) * anchors) * stride
        
        pred_xywh = tf.concat([pred_xy, pred_hw], axis=-1)
        
        pred_conf = tf.sigmoid(conv_bbox_raw_conf)
        
        pred_prob = tf.sigmoid(conv_bbox_raw_prob)
        
        result = tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
        
        return result
    
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
    
    def focal(self, target, actual):
        
        focal_loss = tf.pow(tf.abs(target - actual), 2)
        
        return focal_loss
    
    def giou(self, bbox1, bbox2):
        
        iou, union_area = self.iou(bbox1, bbox2 )
        
        bbox1_cords = tf.concat([bbox1[:2] - bbox1[2:] * 0.5, bbox1[:2] + bbox1[2:] * 0.5], axis=-1)
        bbox2_cords = tf.concat([bbox2[:2] - bbox2[2:] * 0.5, bbox2[:2] + bbox2[2:] * 0.5], axis=-1)
        
        bbox1_cords = tf.concat([tf.minimum(bbox1_cords[:2], bbox1_cords[2:]), 
                                 tf.maximum(bbox1_cords[:2], bbox1_cords[2:])], axis=-1)
        
        bbox2_cords = tf.concat([tf.minimum(bbox2_cords[:2], bbox2_cords[2:]), 
                                 tf.maximum(bbox2_cords[:2], bbox2_cords[2:])], axis=-1)
        
        enclosed_left_up = tf.minimum(bbox1_cords[:2], bbox2_cords[:2])
        enclosed_right_bottom = tf.maximum(bbox1_cords[2:], bbox2_cords[2:])
        
        enclosed_cords = tf.maximum((enclosed_right_bottom - enclosed_left_up), 0.0)
        enclosed_area = enclosed_cords[0] * enclosed_cords[1]
        
        giou = iou - 1.0 * (enclosed_area - union_area)/(enclosed_area + 1e-10)
        
        return giou
    
    def loss_per_scale (self, conv_out, prediction, label, bbox, stride):
        
        conv_out_shape = tf.shape(conv_out)
        batch_size = conv_out_shape[0]
        output_size = conv_out_shape[1]
        input_size = output_size * stride
        conv_out = tf.reshape(conv_out, (batch_size, output_size, output_size, self.anchor_per_scale, self.num_class+5))
        
        raw_conf_score = conv_out[:,:,:,:,4]
        raw_proba_score = conv_out[:,:,:,:,5:]
        
        true_bbox_xywh = label[:,:,:,:,0:4]
        true_conf_score = label[:,:,:,:,4]  # true_conf_score = 1 if there is a object othewise true_conf_score=0
        true_proba_score = label[:,:,:,:,5:]
        
        pred_bbox_xywh = prediction[:,:,:,:,0:4]
        pred_conf_score = prediction[:,:,:,:,4]
        pred_proba_score = prediction[:,:,:,:,5:]
        
        giou = self.giou(true_bbox_xywh, pred_bbox_xywh)
        giou = tf.expand_dims(giou, axis=-1)
        
        bbox_loss_scale = 2.0 - (1.0 * true_bbox_xywh[:,:,:,:,2] * true_bbox_xywh[:,:,:,:,3])/(input_size**2)
        
        giou_loss = true_conf_score * bbox_loss_scale * (1.0 - giou)
        
        iou = self.iou(pred_bbox_xywh[:, :, :, : ,tf.newaxis, :], bbox[:, :, tf.newaxis, tf.newaxis, tf.newaxis, :])
        max_iou = tf.reduce_max(iou, axis=-1)
        max_iou = tf.expand_dims(max_iou, axis=-1)
        
        backgound_detection = (1.0 - true_conf_score) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)
        
        conf_focal = self.focal(true_conf_score, pred_conf_score)
        
        conf_loss = conf_focal * (true_conf_score * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_conf_score, logits=raw_conf_score)
                                  + backgound_detection * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_conf_score, logits=raw_conf_score))
        
        proba_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_proba_score, logits=raw_proba_score)
        
        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        
        proba_loss = tf.reduce_mean(tf.reduce_sum(proba_loss, axis=[1,2,3,4]))
        
        return conf_loss, giou_loss, proba_loss
    
    
    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):
        
        sbbox_loss = self.loss_per_scale(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox, self.strides[0])
        
        mbbox_loss = self.loss_per_scale(self.conve_mbbox, self.pred_mbbox, label_mbbox, true_mbbox, self.strides[1])
        
        lbbox_loss = self.loss_per_scale(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox, self.strides[2])  
        
        giou_loss = sbbox_loss[0] + mbbox_loss[0] + lbbox_loss[0]
        
        conf_loss = sbbox_loss[1] + mbbox_loss[1] + lbbox_loss[1]
        
        proba_loss = sbbox_loss[2] + mbbox_loss[2] + lbbox_loss[2]
        
        return giou_loss, conf_loss, proba_loss