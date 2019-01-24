# -*- coding: utf-8 -*-

from skimage import io,transform, color
from collections import Counter
import glob
import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import vgg

#数据集地址
path=['./dangerous', './safe']
path_v=['./dangerous_v', './safe_v']
path_p=['./dangerous_p', './safe_p']
#模型保存地址
model_path='./model/model.ckpt'

# width / height ratio in 1280*800 image only
# image_p_ws = 180
# image_p_we = 380
# image_p_hs = 310
# image_p_he = 830
# same image width / height ratio in 1280*800 image as that in 640*800 on 2019/1/3
# image_p_ws = 180
# image_p_we = 410
# image_p_hs = 310
# image_p_he = 970
# resize_flag = True
# width / height ratio in 640*480 image
image_p_ws = 90
image_p_we = 230
image_p_hs = 120
image_p_he = 450
resize_flag = True
#将所有的图片resize成100*100
if resize_flag:
    w=100
    h=100
    c=3
else:
    w = image_p_we - image_p_ws
    h = image_p_he - image_p_hs
    c = 3


#读取图片
def read_img(path):
    # cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    cate = path
    imgs=[]
    labels=[]
    mean_v = []
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.bmp'):
            print('reading the images:%s'%(im))
            img_raw=io.imread(im)
            img = img_raw[70:300, 130:600,:]
            img_resize = transform.resize(img,(w,h))
            most_common_n = 1
            mc = Counter(np.reshape(img_resize,(-1))).most_common(most_common_n)
            most_show_value = mc[0][0]
            mask = img_resize == most_show_value
            mask = np.where(img_resize == most_show_value)
            img_resize[mask[0], mask[1], :] = 0
            imgs.append(img_resize)
            labels.append(idx)
            mean_v.append(np.mean(img_resize))
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32), np.asarray(mean_v,np.float32)


def read_img_v(path):
    # cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    cate = path
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.bmp'):
            print('reading the images:%s'%(im))
            img_raw=io.imread(im)
            img = np.append(img_raw[60:170, 120:590,:],img_raw[330:440, 120:590, :], axis=0)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)


def read_img_p(path, resize_flag):
    # cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    cate = path
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.bmp'):
            print('reading the images:%s'%(im))
            img_raw=io.imread(im)
            img = img_raw[image_p_ws:image_p_we, image_p_hs:image_p_he, :]
            # img=transform.resize(img,(w,h))
            img = transform.resize(img,(w,h)) if resize_flag == True else img
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)


def mean_value_of_image(path):
    mean_v = []
    labels = []
    for idx, folder in enumerate(path):
        for im in glob.glob(folder + '/*.bmp'):
            img_raw = io.imread(im)
            img_array = np.append(img_raw[60:170, 120:590, :], img_raw[330:440, 120:590, :], axis=0)
            img_array = transform.resize(img_array, (w,h))
            mean_v.append(img_array.mean())
    return np.asarray(mean_v, np.float32)


# data,label, mean_value=read_img(path)
# data,label=read_img_v(path_v)
data,label=read_img_p(path_p, resize_flag)


#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]


#将所有数据分为训练集和验证集
ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

#-----------------构建网络----------------------
#占位符
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding='VALID')

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight",[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight",[3,3,128,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool4_shape = pool4.get_shape().as_list()
        nodes = pool4_shape[1] * pool4_shape[2] * pool4_shape[3]
        # nodes = 6*6*128
        reshaped = tf.reshape(pool4,[-1,nodes])

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit

#---------------------------网络结束---------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
# logits = inference(x,False,regularizer)
logits = vgg.inference(x,False,regularizer)

#(小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval')

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


#训练和测试数据，可将n_epoch设置更大一些

n_epoch=10
batch_size=80
saver=tf.train.Saver()
sess=tf.Session()
sess.run(tf.global_variables_initializer())
train_accuracy = []
validation_accuracy = []
for epoch in range(n_epoch):
    start_time = time.time()

    print("====epoch %d====="%epoch)

    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
    print("   train acc: %f" % (np.sum(train_acc)/ n_batch))
    train_accuracy.append(np.sum(train_acc) / n_batch)

    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss)/ n_batch))
    print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))

    end_time = time.time()
    print("  epoch time is %f" % (end_time - start_time))
    validation_accuracy.append(np.sum(val_acc) / n_batch)


result_filename = str(time.time())
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(train_accuracy, 'ko-', label = 'train accuracy')
plt.plot(validation_accuracy, 'ko--', label = 'validation accuracy')
ax.set_xlabel('epoch')
plt.legend(loc='best')
plt.savefig(result_filename + '.png', format='png')
plt.close()
np.savez(result_filename, train_accuracy = train_accuracy, validation_accuracy = validation_accuracy)
saver.save(sess,model_path)
sess.close()