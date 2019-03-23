# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path
from os import listdir
from os.path import isfile, join
from random import shuffle
import xml.etree.ElementTree as ET
import pickle
import time

#https://github.com/uoip/transforms
from transforms import *
from numpy.random import RandomState

PRNG = RandomState()
transform = Compose([           
    [ColorJitter(prob=0.5)],
    BoxesToCoords(relative=False),
    HorizontalFlip(),
    Expand((1, 4), prob=0.5),
    ObjectRandomCrop(),
    Resize(300),
    CoordsToBoxes(relative=False),
    ], 
    PRNG, 
    mode='linear', 
    border='constant', 
    fillval=0, 
    outside_points='clamp')

image_width, image_height, C = 416, 416, 3
number_anchors = 5
priors = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),(7.88282, 3.52778), (9.77052, 9.16828)]

number_classes = 20
grid_size = 32

number_anchors == len(priors)
image_height % grid_size == 0
number_grids = int(image_height / grid_size)

annotation_path = "/gpu_data/pkabara/data/object_detection/VOC/VOCdevkit/VOC2007_2012/Annotations/"
image_path = "/gpu_data/pkabara/data/object_detection/VOC/VOCdevkit/VOC2007_2012/JPEGImages/"
file_names_cache = "/gpu_data/pkabara/data/object_detection/VOC/VOCdevkit/VOC2007_2012/names.txt"

classes = {
'class_to_index': {'sheep': 0,  'bird': 1,  'dog': 2,  'motorbike': 3,  'diningtable': 4,  'sofa': 5,  'bicycle': 6,  'bottle': 7,  'boat': 8,  'train': 9,  'cat': 10,  'car': 11,  'cow': 12,  'aeroplane': 13,  'pottedplant': 14,  'tvmonitor': 15,  'bus': 16,  'person': 17,  'chair': 18,  'horse': 19}, 
'index_to_class': {0: 'sheep',  1: 'bird',  2: 'dog',  3: 'motorbike',  4: 'diningtable',  5: 'sofa',  6: 'bicycle',  7: 'bottle',  8: 'boat',  9: 'train',  10: 'cat',  11: 'car',  12: 'cow',  13: 'aeroplane',  14: 'pottedplant',  15: 'tvmonitor',  16: 'bus',  17: 'person',  18: 'chair',  19: 'horse'}
}

class dataset:
    def __init__(self,batch_size=32,apply_transformation=True):
        if os.path.isfile(file_names_cache):
            print ("files names loading from the cache ... ")
            with open(file_names_cache,"rb") as f:
                self.file_names = pickle.load(f)
        else:
            print ("files names loading from the system and caching ... ")
            self.file_names = [f for f in listdir(annotation_path) if isfile(join(annotation_path, f))]
            with open(file_names_cache,"wb") as f:
                pickle.dump(self.file_names,f)
                
        self.len_file_names = len(self.file_names)
        self.count = 0
        self.batch_size = batch_size
        self.prior_boxes = []
        for each in priors:
            self.prior_boxes.append([0,0,each[0],each[1]])
        self.prior_boxes = np.array(self.prior_boxes)
        self.apply_transformation = apply_transformation
    
    def get_bbox(self,ann_file_name):
        bboxes = []
        class_labels = []
        et = ET.parse(ann_file_name)
        # print(ET.tostring(et.getroot(), encoding='utf8').decode('utf8'))
        for child in et.getroot():
            if child.tag == 'object':
                xmin = float(child.find("bndbox").find("xmin").text)
                ymin = float(child.find("bndbox").find("ymin").text)
                xmax = float(child.find("bndbox").find("xmax").text)
                ymax = float(child.find("bndbox").find("ymax").text)
                class_name = child.find("name").text
                class_name = classes['class_to_index'][class_name]
                bboxes.append([xmin,ymin,xmax,ymax])
                class_labels.append(class_name)
                
        return class_labels, bboxes
    
    def get_image(self,image_file_name):
        I = cv2.imread(image_file_name)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        return I
    
    def iou_with_prior(self,zero_shifted_box):
        left_common = np.maximum(self.prior_boxes[:,:2],zero_shifted_box[:2])
        right_common = np.minimum(self.prior_boxes[:,2:],zero_shifted_box[2:])
        common_area = (right_common[:,0] - left_common[:,0]) * (right_common[:,1] - left_common[:,1])
        prior_area = self.prior_boxes[:,2] * self.prior_boxes[:,3] 
        pred_area = zero_shifted_box[2] * zero_shifted_box[3]
        ious = common_area / (prior_area+pred_area-common_area)
        max_iou_index = np.argmax(ious)
        return max_iou_index
    
    def grid_based_processing(self,xc,yc,wid,hei):
        # for prior divide by grid size 
        # prior are interms of grids
        g_xc = xc / grid_size
        g_yc = yc / grid_size
        g_wid = wid / grid_size
        g_hei = hei / grid_size
        
        center_belonging_grid = [int(np.floor(g_xc)),int(np.floor(g_yc))]
        wh_grid = [g_wid,g_hei]
        offset_left_top_of_grid = [g_xc - np.floor(g_xc), g_yc - np.floor(g_yc)]
        
        # to get best matching prior
        zero_shifted_box = np.array([0,0,g_wid,g_hei])
        # check best fitting prior
        best_prior_index = self.iou_with_prior(zero_shifted_box)
        
        return center_belonging_grid, offset_left_top_of_grid, wh_grid, best_prior_index
    
    def label_processing(self,image,class_labels,bboxes):
        # label => xyhwc between 0 to 1 (normalized by image size)
        # mask => which grid, anchor conatins image
        # offset => at max iou matching anchor in particular grid cell offset (between 0-1) 
        label = [] # number_boxes x 5(xywhc)
        mask = np.zeros([number_grids,number_grids,number_anchors])
        offset_to_grid_cell = np.zeros([number_grids,number_grids,number_anchors,5])
        
        H,W,C = image.shape
        image = cv2.resize(image,(image_width, image_height))
        # image normalization
        image = ((image / 255.0) * 2.0) - 1.0
        new_width_ratio = image_width / float(W)
        new_height_ratio = image_height / float(H)
        
        for each_box, each_class in zip(bboxes,class_labels):
            # image boxes mapped to new shape
            xmin,ymin,xmax,ymax = each_box
            xmin,ymin = xmin*new_width_ratio, ymin*new_height_ratio
            xmax,ymax = xmax*new_width_ratio, ymax*new_height_ratio
            
            xc = xmin + (xmax-xmin)/2.0
            yc = ymin + (ymax-ymin)/2.0
            wid = (xmax-xmin)
            hei = (ymax-ymin)
            
            # check for non zero height 
            # if box out of image because transform
            if wid==0 or hei==0:
                continue
            
            # update mask and offset_to_grid_cell
            (center_belonging_grid, offset_left_top_of_grid, wh_grid, best_prior_index) = self.grid_based_processing(xc,yc,wid,hei)
            x,y = center_belonging_grid
            a = best_prior_index
            mask[y,x,a] = 1
            offset_to_grid_cell[y,x,a] = [offset_left_top_of_grid[0],offset_left_top_of_grid[1],
                                         np.log(wh_grid[0] / priors[best_prior_index][0]),
                                         np.log(wh_grid[1] / priors[best_prior_index][1]),
                                         each_class]
            
            label.append([(center_belonging_grid[0]+offset_left_top_of_grid[0])/ number_grids,
                         (center_belonging_grid[1]+offset_left_top_of_grid[1])/ number_grids,
                          wh_grid[0] / number_grids,
                          wh_grid[1] / number_grids,
                          each_class
                         ])
        
        return image, label, mask, offset_to_grid_cell
            
    def batch(self):
        c_batch = 0 
        images, labels, masks, offset_to_grid_cells = [], [], [], []
        batch_single_image_boxes = 0
        while c_batch < self.batch_size:
            if self.count >= self.len_file_names:
                self.count = 0 
                shuffle(self.file_names)
            
            file_name = self.file_names[self.count]
            annotation_name = annotation_path + file_name
            image_name = image_path + file_name.strip(".xml") + ".jpg"
            
            class_labels, bboxes = self.get_bbox(annotation_name)
            I = self.get_image(image_name)
            
            # apply transformation here ! (During test make it False ...)
            # change class labels if bbox out of image after transform
            if self.apply_transformation :
                I, bboxes = transform(I,bboxes)
            
            if batch_single_image_boxes < len(bboxes):
                batch_single_image_boxes = len(bboxes)
            
            image, label, mask, offset_to_grid_cell = self.label_processing(I,class_labels,bboxes)
            images.append(image)
            labels.append(label)
            masks.append(mask)
            offset_to_grid_cells.append(offset_to_grid_cell)
            
            c_batch += 1
            self.count += 1
            
        # pad with zeros to make bboxes same for all images
        padded_labels = []
        for each in labels:
            each = np.array(each)
            padding = np.zeros([batch_single_image_boxes - len(each),5])
            each = np.concatenate([each,padding],axis=0)
            padded_labels.append(each)
        
        return np.array(images), np.array(padded_labels), np.array(masks), np.array(offset_to_grid_cells)
