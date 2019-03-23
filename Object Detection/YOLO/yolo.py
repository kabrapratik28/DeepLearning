import numpy as np
import tensorflow as tf
tf.reset_default_graph()

image_width, image_height, C = 416, 416, 3
number_anchors = 5
priors = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),(7.88282, 3.52778), (9.77052, 9.16828)]

number_classes = 20
grid_size = 32
number_anchors == len(priors)
image_height % grid_size == 0
number_grids = int(image_height / grid_size)

no_obj_const = 0.5
obj_const = 4.0
coord_const  = 5.0

model_save_path = "/gpu_data/pkabara/data/object_detection/yolo_model/yolo"

# testing
max_output_size=15

class yolo():
    def __init__(self):
        self.input = tf.placeholder(dtype=tf.float32,shape=[None,image_height, image_width, C])
        self.is_training = tf.placeholder(dtype=tf.bool)
        
        # label => xyhwc between 0 to 1 (normalized by image size)
        # mask => which grid, anchor conatins image
        # offset => at max iou matching anchor in particular grid cell offset (between 0-1) 
        self.label = tf.placeholder(dtype=tf.float32,shape=[None,None,5]) # B x number_boxes x 5(xywhc)
        self.mask = tf.placeholder(dtype=tf.float32,shape=[None,number_grids,number_grids,number_anchors])
        self.offset_to_grid_cell = tf.placeholder(dtype=tf.float32,shape=[None,number_grids,number_grids,number_anchors,5])
        
        self.output = None
        self.loss = None
        self.merged_summary = None
        
        self.model()
        self.calc_loss()

        # testing
        self.cutoff_prob = tf.placeholder(dtype=tf.float32,shape=[1])
        self.output_class_idx = None
        self.output_borders = None
        self.output_proba = None
        self.test()
                
    def leaky_relu_modified_alpha(self,x):
        return tf.nn.leaky_relu(x,alpha=0.1)
    
    def conv(self,x, filters, kernel_size, strides, is_training, activation, name):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(x,filters,kernel_size,strides,activation=activation ,padding='same',kernel_initializer=tf.glorot_normal_initializer(),name=name)
            x = tf.layers.batch_normalization(x,training=is_training)
        print (x)
        return x
    
    def maxpool(self, x, pool_size, strides, name):
        with tf.variable_scope(name):
            x = tf.layers.max_pooling2d(x, pool_size, strides, padding='same', name=name)
        print (x)
        return x
    
    def concat(self,x,y):
        y = tf.space_to_depth(y,2)
        c = tf.concat([x,y],axis=-1)
        return c
            
    def model(self):
        x = self.conv(self.input, 32, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv1')
        x = self.conv(x, 32, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv2')
        x = self.maxpool(x, (2,2), (2,2), 'maxpool1')
        
        x = self.conv(x, 64, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv3')
        x = self.maxpool(x, (2,2), (2,2), 'maxpool2')
        
        x = self.conv(x, 128, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv4')
        x = self.conv(x, 64, (1,1), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv5')
        x = self.conv(x, 128, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv6')
        x = self.maxpool(x, (2,2), (2,2), 'maxpool3')
        
        x = self.conv(x, 256, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv7')
        x = self.conv(x, 128, (1,1), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv8')
        x = self.conv(x, 256, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv9')
        x = self.maxpool(x, (2,2), (2,2), 'maxpool4')
        
        x = self.conv(x, 512, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv10')
        x = self.conv(x, 256, (1,1), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv11')
        x = self.conv(x, 512, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv12')
        x = self.conv(x, 256, (1,1), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv13')
        passthrough = self.conv(x, 512, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv14')  
        x = self.maxpool(passthrough, (2,2), (2,2), 'maxpool5')
        
        x = self.conv(x, 1024, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv15')
        x = self.conv(x, 512, (1,1), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv16')
        x = self.conv(x, 1024, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv17')
        x = self.conv(x, 512, (1,1), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv18')
        x = self.conv(x, 1024, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv19')
        
        x = self.conv(x, 1024, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv20')
        x = self.conv(x, 1024, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv21')
        x = self.concat(x,passthrough)
        x = self.conv(x, 1024, (3,3), 1, self.is_training, self.leaky_relu_modified_alpha, 'conv22')
        
        x = self.conv(x, number_anchors * (number_classes + 5), (1,1), 1, self.is_training, None, 'conv23')
        x = tf.reshape(x,(-1,number_grids,number_grids,number_anchors,5+number_classes))
        
        self.output = x
        print (self.output)

    def test(self):
        predict = self.output
        
        # apply sigmoid to model prediction
        xy = tf.sigmoid(predict[...,:2]) # B x G x G x A x 2
        wh_b = predict[...,2:4]
        wh = tf.exp(wh_b) * np.array(priors) # B x G x G x A x 2 # in terms of number of grids !
        objectness = tf.sigmoid(predict[...,4]) # B x G x G x A
        classes = tf.nn.softmax(predict[...,5:]) # B x G x G x A x C

        # apply offset xy, wh
        p = tf.reshape(tf.tile(tf.range(0,number_grids),[number_grids]),[1,number_grids,number_grids,1,1])
        q = tf.transpose(p,[0,2,1,3,4])
        offset = tf.concat([p,q],axis=-1)
        offset = tf.to_float(tf.tile(offset,[1,1,1,number_anchors,1])) # B x G x G x A x 2
        center_xy = xy + offset # B x G x G x A x 2
        
        # center_xy and wh are in terms of grids
        # multiple by grid size made them in terms of image size
        center_xy = center_xy * grid_size
        wh = wh * grid_size

        left_top = center_xy - wh / 2.0
        right_bottom = center_xy + wh / 2.0
        borders = tf.concat([left_top,right_bottom],axis=-1) # B x G x G x A x 4

        # highly probable class and its probability
        max_prob_class = tf.reduce_max(classes,axis=-1) # B x G x G x A 
        max_prob_class_idx = tf.argmax(classes,axis=-1) # B x G x G x A 
        probability = objectness * max_prob_class # B x G x G x A 

        max_prob_class_idx = tf.reshape(max_prob_class_idx,shape=[-1,number_grids*number_grids*number_anchors])
        probability = tf.reshape(probability,shape=[-1,number_grids*number_grids*number_anchors])
        borders = tf.reshape(borders,shape=[-1,number_grids*number_grids*number_anchors,4])

        score_threshold = tf.cast(probability >= self.cutoff_prob,dtype=tf.float32)
        probability = probability * score_threshold
        
        self.output_class_idx = max_prob_class_idx
        self.output_borders = borders
        self.output_proba = probability

    def calc_loss(self):
        predict = self.output
        label = self.label # B x number_boxes x 5(xywhc)
        mask = self.mask # B x G x G x A
        offset_to_grid_cell = self.offset_to_grid_cell # B x G x G x A x 5
        
        batch_size = tf.cast(tf.shape(label)[0],dtype=tf.float32)
        number_boxes = tf.shape(label)[0]
        
        # apply sigmoid to model prediction
        xy = tf.sigmoid(predict[...,:2]) # B x G x G x A x 2
        wh_b = predict[...,2:4]
        wh = tf.exp(wh_b) * np.array(priors) # B x G x G x A x 2 # in terms of number of grids !
        objectness = tf.sigmoid(predict[...,4]) # B x G x G x A
        classes = tf.nn.softmax(predict[...,5:]) # B x G x G x A x C
        
        # apply offset xy, wh
        p = tf.reshape(tf.tile(tf.range(0,number_grids),[number_grids]),[1,number_grids,number_grids,1,1])
        q = tf.transpose(p,[0,2,1,3,4])
        offset = tf.concat([p,q],axis=-1)
        offset = tf.to_float(tf.tile(offset,[1,1,1,number_anchors,1])) # B x G x G x A x 2
        center_xy = xy + offset # B x G x G x A x 2
        
        # center_xy, wh in terms of number_of_grids
        # center_xy, wh convert between 0-1
        center_xy = center_xy / number_grids
        wh = wh / number_grids
        center_xywh = tf.concat([center_xy,wh],axis=-1) # B x G x G x A x 4
        center_xywh = tf.expand_dims(center_xywh,4) # B x G x G x A x 1 x 4
        
        label_shape = tf.shape(label) # B x number_boxes x 5
        # reshape for each grid and anchor # B x G x G x A x number_boxes x 5
        label = tf.reshape(label,[label_shape[0],1,1,1,label_shape[1],label_shape[2]]) # B x 1 x 1 x 1 x number_boxes x 5
        
        # iou calculate between center_xywh, label
        p_left_xy = center_xywh[...,:2] - center_xywh[...,2:4] / 2.0
        p_right_xy = center_xywh[...,:2] + center_xywh[...,2:4] / 2.0
        
        g_left_xy = label[...,:2] - label[...,2:4] / 2.0
        g_right_xy = label[...,:2] + label[...,2:4] / 2.0
        
        left_common = tf.maximum(p_left_xy, g_left_xy)
        right_common = tf.minimum(p_right_xy, g_right_xy)
        area_common = (right_common[...,0] - left_common[...,0]) * (right_common[...,1] - left_common[...,1])
        
        p_area = center_xywh[...,2] * center_xywh[...,3]
        g_area = label[...,2] * label[...,3]
        
        ious = area_common / (p_area + g_area - area_common) # B x G x G x A x number_boxes
        ious = tf.maximum(0.0,tf.minimum(1.0,ious))
        best_ious = tf.reduce_max(ious,axis=-1) # B x G x G x A
        mask_obj_area_covers =  tf.cast(best_ious > 0.6,dtype=tf.float32)
        
        # confidence loss
        noobj_loss = no_obj_const * (1 - mask_obj_area_covers) * (1 - mask) * tf.square(0.0 - objectness)
        obj_loss = obj_const * mask * tf.square(1.0 - objectness)
        confidence_loss = obj_loss + noobj_loss
        
        # class loss
        ext_mask = tf.expand_dims(mask,axis=-1)
        object_classes = tf.cast(offset_to_grid_cell[...,4],dtype=tf.int32) # B x G x G x A
        ohe_object_classes = tf.one_hot(object_classes,number_classes) # B x G x G x A x C
        class_loss = ext_mask * tf.square(ohe_object_classes - classes)
        
        # coordinate loss
        xy_loss = ext_mask * tf.square(offset_to_grid_cell[...,:2] - xy) # B x G x G x A x 2
        wh_loss = ext_mask * tf.square(offset_to_grid_cell[...,2:4] - wh_b) # B x G x G x A x 2
        coord_loss = xy_loss + wh_loss
        
        total_confidence_loss = tf.reduce_sum(confidence_loss) / batch_size
        total_class_loss = tf.reduce_sum(class_loss) / batch_size
        total_coord_loss = tf.reduce_sum(coord_loss) / batch_size
        
        total_loss = (total_confidence_loss + total_class_loss + total_coord_loss)
        self.loss = total_loss
        
        tf.summary.scalar("total_confidence_loss",total_confidence_loss)
        tf.summary.scalar("total_class_loss",total_class_loss)
        tf.summary.scalar("total_coord_loss",total_coord_loss)
        tf.summary.scalar("total_loss",self.loss)
        tf.summary.histogram("ious",ious)
        self.merged_summary = tf.summary.merge_all()
        
    def save_model(self,sess):
        saver = tf.train.Saver()
        saver.save(sess, model_save_path)
    
    def load_model(self,sess):
        saver = tf.train.Saver()
        if tf.train.checkpoint_exists(model_save_path):
            print ("restoring model present in checkpoint ... ")
            saver.restore(sess, model_save_path)