import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
from dataset import dataset
from yolo import yolo

logs_path = "/gpu_data/pkabara/data/object_detection/logs"

def train(yolo,dataset,max_iteration,save_model_each_iter=1000):
        
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        optimizer = tf.train.AdamOptimizer(1e-4)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(yolo.loss)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # if model present restore it!
        yolo.load_model(sess)

        for each_iter in range(max_iteration):
                images, labels, masks, offset_to_grid_cells = dataset.batch()

                _, summary, loss = sess.run([train_op,yolo.merged_summary,yolo.loss],
                                    feed_dict={
                                                yolo.is_training:True,
                                                yolo.input : images,
                                                yolo.label : labels,
                                                yolo.mask : masks,
                                                yolo.offset_to_grid_cell : offset_to_grid_cells
                                               })
                
                summary_writer.add_summary(summary, each_iter)
                
                # Save model after sometimes !
                if each_iter % save_model_each_iter == 0:
                    yolo.save_model(sess)
                    
                # print on the console !
                if each_iter % 20 == 0:
                    print ("each iter "+str(each_iter)+" loss "+ str(loss))
                
        print ("Done !")

if __name__ == "__main__":
    dataset = dataset()
    yolo = yolo()
    train(yolo=yolo, dataset=dataset, max_iteration=100000)