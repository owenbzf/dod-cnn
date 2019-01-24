# -*- coding: utf-8 -*-

from skimage import io,transform, color
from collections import Counter
import glob
import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights1 = tf.get_variable("weight1_1",[3,1,3,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_weights2 = tf.get_variable("weight1_2",[3,1,64,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases1 = tf.get_variable("bias1_1", [64], initializer=tf.constant_initializer(0.0))
        conv1_biases2 = tf.get_variable("bias1_2", [64], initializer=tf.constant_initializer(0.0))
        conv1_1 = tf.nn.conv2d(input_tensor, conv1_weights1, strides=[1, 1, 1, 1], padding='SAME')
        relu1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1, conv1_biases1))
        conv1_2 = tf.nn.conv2d(relu1_1, conv1_weights2, strides=[1, 1, 1, 1], padding='SAME')
        relu1_2 = tf.nn.relu(tf.nn.bias_add(conv1_2, conv1_biases2))
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1_2, ksize = [1,2,2,1],strides=[1,2,2,1],padding='VALID')

    with tf.variable_scope("layer3-conv2"):
        conv2_weights1 = tf.get_variable("weight2_1",[3,1,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_weights2 = tf.get_variable("weight2_2",[3,1,128,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases1 = tf.get_variable("bias2_1", [128], initializer=tf.constant_initializer(0.0))
        conv2_biases2 = tf.get_variable("bias2_2", [128], initializer=tf.constant_initializer(0.0))
        conv2_1 = tf.nn.conv2d(pool1, conv2_weights1, strides=[1, 1, 1, 1], padding='SAME')
        relu2_1 = tf.nn.relu(tf.nn.bias_add(conv2_1, conv2_biases1))
        conv2_2 = tf.nn.conv2d(relu2_1, conv2_weights2, strides=[1, 1, 1, 1], padding='SAME')
        relu2_2 = tf.nn.relu(tf.nn.bias_add(conv2_2, conv2_biases2))
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer5-conv3"):
        conv3_weights1 = tf.get_variable("weight3_1",[3,1,128,256],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_weights2 = tf.get_variable("weight3_2",[3,1,256,256],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #conv3_weights3 = tf.get_variable("weight3_3",[3,3,256,256],initializer=tf.truncated_normal_initializer(stddev=0))
        conv3_biases1 = tf.get_variable("bias3_1", [256], initializer=tf.constant_initializer(0.0))
        conv3_biases2 = tf.get_variable("bias3_2", [256], initializer=tf.constant_initializer(0.0))
        #conv3_biases3 = tf.get_variable("bias3_3", [256], initializer=tf.constant_initializer(0.0))
        conv3_1 = tf.nn.conv2d(pool2, conv3_weights1, strides=[1, 1, 1, 1], padding='SAME')
        relu3_1 = tf.nn.relu(tf.nn.bias_add(conv3_1, conv3_biases1))
        conv3_2 = tf.nn.conv2d(relu3_1, conv3_weights2, strides=[1, 1, 1, 1], padding='SAME')
        relu3_2 = tf.nn.relu(tf.nn.bias_add(conv3_2, conv3_biases2))
        #conv3_3 = tf.nn.conv2d(relu3_2, conv3_weights3, strides=[1, 1, 1, 1], padding='SAME')
        #relu3_3 = tf.nn.relu(tf.nn.bias_add(conv3_3, conv3_biases3))
    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer7-conv4"):
        conv4_weights1 = tf.get_variable("weight4_1",[3,1,256,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_weights2 = tf.get_variable("weight4_2",[3,1,512,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #conv4_weights3 = tf.get_variable("weight4_3",[3,3,512,512],initializer=tf.truncated_normal_initializer(stddev=0))
        conv4_biases1 = tf.get_variable("bias4_1", [512], initializer=tf.constant_initializer(0.0))
        conv4_biases2 = tf.get_variable("bias4_2", [512], initializer=tf.constant_initializer(0.0))
        #conv4_biases3 = tf.get_variable("bias4_3", [512], initializer=tf.constant_initializer(0.0))
        conv4_1 = tf.nn.conv2d(pool3, conv4_weights1, strides=[1, 1, 1, 1], padding='SAME')
        relu4_1 = tf.nn.relu(tf.nn.bias_add(conv4_1, conv4_biases1))
        conv4_2 = tf.nn.conv2d(relu4_1, conv4_weights2, strides=[1, 1, 1, 1], padding='SAME')
        relu4_2 = tf.nn.relu(tf.nn.bias_add(conv4_2, conv4_biases2))
        #conv4_3 = tf.nn.conv2d(relu4_2, conv4_weights3, strides=[1, 1, 1, 1], padding='SAME')
        #relu4_3 = tf.nn.relu(tf.nn.bias_add(conv4_3, conv4_biases3))
    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer9-conv5"):
        conv5_weights1 = tf.get_variable("weight5_1",[3,1,512,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_weights2 = tf.get_variable("weight5_2",[3,1,512,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #conv5_weights3 = tf.get_variable("weight5_3",[3,3,512,512],initializer=tf.truncated_normal_initializer(stddev=0))
        conv5_biases1 = tf.get_variable("bias5_1", [512], initializer=tf.constant_initializer(0.0))
        conv5_biases2 = tf.get_variable("bias5_2", [512], initializer=tf.constant_initializer(0.0))
        #conv5_biases3 = tf.get_variable("bias5_3", [512], initializer=tf.constant_initializer(0.0))
        conv5_1 = tf.nn.conv2d(pool4, conv5_weights1, strides=[1, 1, 1, 1], padding='SAME')
        relu5_1 = tf.nn.relu(tf.nn.bias_add(conv5_1, conv5_biases1))
        conv5_2 = tf.nn.conv2d(relu5_1, conv5_weights2, strides=[1, 1, 1, 1], padding='SAME')
        relu5_2 = tf.nn.relu(tf.nn.bias_add(conv5_2, conv5_biases2))
        #conv5_3 = tf.nn.conv2d(relu5_2, conv5_weights3, strides=[1, 1, 1, 1], padding='SAME')
        #relu5_3 = tf.nn.relu(tf.nn.bias_add(conv5_3, conv5_biases3))
    with tf.name_scope("layer10-pool5"):
        pool5 = tf.nn.max_pool(relu5_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool5_shape = pool5.get_shape().as_list()
        nodes = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
        # nodes = 6*6*128
        print("a=%d,b=%d,c=%d"%(pool5_shape[1],pool5_shape[2],pool5_shape[3]))
        reshaped = tf.reshape(pool5,[-1,nodes])

    with tf.variable_scope('layer11-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 4096],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias11", [4096], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    #with tf.variable_scope('layer12-fc2'):
        #fc2_weights = tf.get_variable("weight12", [4096, 4096],
         #                             initializer=tf.truncated_normal_initializer(stddev=0))
        #if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        #fc2_biases = tf.get_variable("bias12", [4096], initializer=tf.constant_initializer(0))

        #fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        #if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer13-fc3'):
        fc3_weights = tf.get_variable("weight13", [4096, 2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias13", [2], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc3_weights) + fc3_biases

    return logit

