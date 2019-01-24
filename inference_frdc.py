# -*- coding: utf-8 -*-

from skimage import io,transform
import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
import time

path1 = './ti.1.bin.bmp'
test_file_path = './test_file'
test_file_path_v = './test_file_v'
test_file_path_p = './wenxinyu/20181219'

face_dict = {1:'safe',0:'dangerous'}

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

def read_one_image(path):
    img_raw = io.imread(path)
    img = img_raw[70:300, 130:600, :]
    img = transform.resize(img,(w,h))
    return np.asarray(img)


def read_image(path):
    cate = path
    imgs=[]
    file_name = []
    for im in glob.glob(path+'/*.bmp'):
        print('reading the images:%s'%(im))
        img_raw=io.imread(im)
        img = img_raw[70:300, 130:600,:]
        img=transform.resize(img,(w,h))
        imgs.append(img)
        file_name.append(im)
    return np.asarray(imgs), file_name


def read_image_v(path):
    imgs=[]
    file_name = []
    for im in glob.glob(path+'/*.bmp'):
        print('reading the images:%s'%(im))
        img_raw=io.imread(im)
        img = np.append(img_raw[60:170, 120:590, :], img_raw[330:440, 120:590, :], axis=0)
        img=transform.resize(img,(w,h))
        imgs.append(img)
        file_name.append(im)
    return np.asarray(imgs), file_name


def read_image_p(path):
    imgs=[]
    file_name = []
    for im in glob.glob(path+'/*.bmp'):
        print('reading the images:%s'%(im))
        img_raw=io.imread(im)
        file_index = im[len(path)+1:-4]
        img = img_raw[image_p_ws:image_p_we, image_p_hs:image_p_he, :]
        img = transform.resize(img, (w, h)) if resize_flag == True else img
        imgs.append(img)
        # file_name.append(im)
        file_name.append(file_index)
    return np.asarray(imgs), file_name


with tf.Session() as sess:
    data = []
    # data, fn= read_image(test_file_path)
    # data, fn= read_image(test_file_path_v)
    data, fn= read_image_p(test_file_path_p)
    # data1 = read_one_image(path1)
    # data.append(data1)

    saver = tf.train.import_meta_graph('./wenxinyu/model/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./wenxinyu/model/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)

    #打印出预测矩阵
    print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result,1).eval())
    #根据索引通过字典对应人脸的分类
    output = []
    output = tf.argmax(classification_result,1).eval()
    fn = list(map(int, fn))
    map_fn = dict(zip(fn, range(len(fn))))
    # map_fn = map_fn[sorted(map_fn.keys())]
    [(k, map_fn[k]) for k in sorted(map_fn.keys())]
    file_name = str('result.'+str(time.time())+'.txt')
    result_file = open(file_name, 'w')
    for i in range(len(output)):
        # print("No.",i+1,fn[i],face_dict[output[i]])
        # print(fn[i], face_dict[output[i]])
        print('%d\t%s' % (i+1, face_dict[output[map_fn[i+1]]]), file=result_file)
    result_file.close()
