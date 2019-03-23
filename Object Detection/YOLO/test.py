import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"

import numpy as np
import tensorflow as tf
from dataset import dataset
from yolo import yolo
import cv2

image_save_location = "/gpu_data/pkabara/data/object_detection/prediction/"
image_names = 0 
classes = {
'class_to_index': {'sheep': 0,  'bird': 1,  'dog': 2,  'motorbike': 3,  'diningtable': 4,  'sofa': 5,  'bicycle': 6,  'bottle': 7,  'boat': 8,  'train': 9,  'cat': 10,  'car': 11,  'cow': 12,  'aeroplane': 13,  'pottedplant': 14,  'tvmonitor': 15,  'bus': 16,  'person': 17,  'chair': 18,  'horse': 19}, 
'index_to_class': {0: 'sheep',  1: 'bird',  2: 'dog',  3: 'motorbike',  4: 'diningtable',  5: 'sofa',  6: 'bicycle',  7: 'bottle',  8: 'boat',  9: 'train',  10: 'cat',  11: 'car',  12: 'cow',  13: 'aeroplane',  14: 'pottedplant',  15: 'tvmonitor',  16: 'bus',  17: 'person',  18: 'chair',  19: 'horse'}
}


def box_ious(cbox,bbox):
    common_lxy = np.maximum(cbox[:2],bbox[:,:2])
    common_rxy = np.minimum(cbox[2:],bbox[:,2:])
    common_area = common_rxy - common_lxy + 1
    common_area = common_area[:,0] * common_area[:,1]
    width = bbox[:,2] - bbox[:,0]
    height = bbox[:,3] - bbox[:,1]
    areas = width * height
    cbox_area = (cbox[2] - cbox[0]) * (cbox[3] - cbox[1])
    
    ious = common_area / (areas + cbox_area - common_area + 1e-6)
    
    return ious
    
def non_max_suppression(bbox,scores,classes,iou_threshold=0.5):
    increasing_scores_index = np.argsort(scores)
    decreasing_scores_index = increasing_scores_index[::-1]
    
    # select according to highest scorer
    bbox_with_highest_prob = np.take(bbox,decreasing_scores_index,axis=0)
    scores = np.take(scores,decreasing_scores_index)
    classes = np.take(classes,decreasing_scores_index)
    
    # compare iou of bboxes with next bboxes only ! As they are decreasing order.
    for i in range(len(bbox_with_highest_prob)):
        cbox = bbox_with_highest_prob[i]
        ious = box_ious(cbox,bbox_with_highest_prob)
        mask = ious > 0.5
        
        not_curr_classes = (classes!=classes[i])
        # masking not in current class and higher probabilities, self (ith ele)
        mask[not_curr_classes] = 0
        mask[:i+1] = 0
        
        mask = 1.0 - mask
        exp_mask = np.expand_dims(mask,axis=1)
        
        bbox_with_highest_prob = bbox_with_highest_prob * exp_mask
        scores = scores * mask
        classes = classes * mask
        
    return bbox_with_highest_prob, scores, classes


def label_image(images, output_class_idx, output_borders, output_proba):
	global image_names

	images = (images + 1.0)/2.0 * 255.0 
	images = images.astype(np.uint8)

	for i in range(len(images)):
		I = images[i]
		idxs = output_class_idx[i]
		borders = output_borders[i]
		proba = output_proba[i]

		# apply non max suppression
		borders, proba, idxs = non_max_suppression(borders,proba,idxs)

		for idx, bbox, p in zip(idxs,borders,proba):
			if p>0:
				lx, ly, rx, ry = map(int,bbox)
				label = classes['index_to_class'][int(idx)] + " | " + str(round(p, 2))
				I = cv2.rectangle(I, (lx, ly), (rx, ry), (0, 255, 0), 2)
				I = cv2.putText(I, str(label), (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), lineType=cv2.LINE_AA)

		I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
		cv2.imwrite(image_save_location + str(image_names) + ".png" ,I)

		image_names += 1


def test(yolo,dataset,max_iteration):
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	# if model present restore it!
	yolo.load_model(sess)

	for each_iter in range(max_iteration):
		images, labels, masks, offset_to_grid_cells = dataset.batch()

		output_class_idx, output_borders, output_proba = sess.run([yolo.output_class_idx, yolo.output_borders, yolo.output_proba],
											feed_dict={
												yolo.is_training:False,
												yolo.input : images,
												yolo.cutoff_prob : np.array([0.6])
											})

		label_image(images, output_class_idx, output_borders, output_proba)


if __name__ == "__main__":
	dataset = dataset(apply_transformation=False)
	yolo = yolo()
	test(yolo=yolo, dataset=dataset, max_iteration=10)