''' AdapNet++:  Self-Supervised Model Adaptation for Multimodal Semantic Segmentation

 Copyright (C) 2018  Abhinav Valada, Rohit Mohan and Wolfram Burgard

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.'''

import argparse
import datetime
import importlib
import os
import numpy as np
import re
import tensorflow as tf
import yaml
from dataset.helper import *
import cv2



PARSER = argparse.ArgumentParser()
PARSER.add_argument('-c', '--config', default='config/cityscapes_train.config')
PARSER.add_argument('-o', '--save_dir', default='/mnt/checkpoints_HDD/IDD/records')



def resize(image,height,width):
   img = np.zeros_like(image)
   for i in range(image.shape[0]):
       img[i] = cv2.resize(image[i],(width,height),interpolation=cv2.INTER_NEAREST)
   return img

def train_func(config,save_dir):
    os.environ['CUDA_VISIBLE_DEVICS'] = config['gpu_id']
    module = importlib.import_module('models.'+config['model'])
    model_func = getattr(module, config['model'])
    data_list, iterator = get_train_data(config)
    #data_list[0] = tf.py_func(func=resize,inp=[data_list[0],384,768],Tout=tf.float32)
    #data_list[1] = tf.py_func(func=resize,inp=[data_list[1],384,768],Tout=tf.float32)
    #print(np.unique(data_list[1]))
    resnet_name = 'resnet_v2_50'
    global_step = tf.Variable(0, trainable=False, name='Global_Step')

    with tf.variable_scope(resnet_name):
        model = model_func(num_classes=config['num_classes'], learning_rate=config['learning_rate'],
                           decay_steps=config['max_iteration'], power=config['power'],
                           global_step=global_step)
        images_pl = tf.placeholder(tf.float32, [None, config['height'], config['width'], 3])
        labels_pl = tf.placeholder(tf.float32, [None, config['height'], config['width'],
                                                config['num_classes']])
        model.build_graph(images_pl, labels_pl)
        # model_vars = tf.trainable_variables()
        # tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        model.create_optimizer()
 
    config1 = tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)
    sess.run(tf.global_variables_initializer())
    step = 0
    total_loss = 0.0
    t0 = None
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(config['checkpoint'],
                                                                      'checkpoint')))
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.Saver(max_to_keep=1000)
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])+1
        sess.run(tf.assign(global_step, step))
        print('Model Loaded')

    else:
        if 'intialize' in config:
            reader = tf.train.NewCheckpointReader(config['intialize'])
            var_str = reader.debug_string()
            print("Var str type :",type(var_str))
            # print('Var is :',str(var_str))
            name_var = re.findall('[A-Za-z0-9/:_]+ ', str(var_str))
            import_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            initialize_variables = {} 
            for var in import_variables: 
                if var.name+' ' in  name_var:
                    initialize_variables[var.name] = var

            saver = tf.train.Saver(initialize_variables)
            saver.restore(save_path=config['intialize'], sess=sess)
            print('Pretrained Intialization')
        saver = tf.train.Saver(max_to_keep=1000)
    writer = tf.summary.FileWriter(save_dir)
    loss_summ = tf.summary.scalar('loss',model.loss)
    image_summ = tf.summary.image('image',data_list[0])
    label_tensor = tf.cast(tf.argmax(data_list[1],axis=3),dtype=tf.uint8)
    label_unresized_tensor = tf.cast(255*data_list[2],dtype=tf.uint8)
    pred_tensor = tf.cast(tf.argmax(model.softmax,3),dtype=tf.uint8)
    label_summ = tf.summary.image('label',255*tf.expand_dims(label_tensor,axis=3))
    #print(tf.shape(label_unresized_tensor))
    label_unresized_summ = tf.summary.image('label_unresized',255*tf.expand_dims(label_unresized_tensor,axis = 3))
    pred_summ = tf.summary.image('pred',255*tf.expand_dims(pred_tensor,axis=3))
    print('Shape of image being summarized: ',tf.argmax(model.softmax,3)[0])
    while 1:
        try:
            img, label,image_str,label_str,summary_str = sess.run([data_list[0], data_list[1],image_summ,label_summ,label_unresized_summ])
            if step==0:
               print("Image size :",img.shape)
            writer.add_summary(image_str,step)
            writer.add_summary(label_str,step)
            writer.add_summary(summary_str,step)
            feed_dict = {images_pl: img, labels_pl: label}
            
            loss_batch, _, loss_str,pred_str = sess.run([model.loss, model.train_op,loss_summ,pred_summ],
                                     feed_dict=feed_dict)
            writer.add_summary(loss_str,step)
            writer.add_summary(pred_str,step)
            total_loss += loss_batch

            if (step + 1) % config['save_step'] == 0:
                saver.save(sess, os.path.join(config['checkpoint'], 'model.ckpt'), step)

            if (step + 1) % config['skip_step'] == 0:
                left_hours = 0

                if t0 is not None:
                    delta_t = (datetime.datetime.now() - t0).seconds
                    left_time = (config['max_iteration'] - step) / config['skip_step'] * delta_t
                    left_hours = left_time/3600.0

                t0 = datetime.datetime.now()
                total_loss /= config['skip_step']
                print('%s %s] Step %s, lr = %f ' \
                  % (str(datetime.datetime.now()), str(os.getpid()), step,
                     model.lr.eval(session=sess)))
                print('\t loss = %.4f' % (total_loss))
                print('\t estimated time left: %.1f hours. %d/%d' % (left_hours, step,
                                                                     config['max_iteration']))
                print('\t', config['model'])
                total_loss = 0.0

            step += 1
            writer.flush()
              
            if step > config['max_iteration']:
                saver.save(sess, os.path.join(config['checkpoint'], 'model.ckpt'), step-1)
                print('training_completed')
                writer.add_graph(sess.graph)
                break
       
        except tf.errors.OutOfRangeError:
            print('Epochs in dataset repeat < max_iteration')
            break

def main():
    args = PARSER.parse_args()
    if args.config:
        file_address = open(args.config)
        config = yaml.load(file_address)
    else:
        print('--config config_file_address missing')
    train_func(config,args.save_dir)

if __name__ == '__main__':
    main()
