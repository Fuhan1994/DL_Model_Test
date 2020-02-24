import tensorflow as tf
import numpy as np
import pdb
from datetime import datetime
from vgg import *

batch_size=1
lr=0.001
num_classes=2
max_steps=50000


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    # 转换为float32类型，并做归一化处理
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int64)
    return img, label

def train():
    x=tf.placeholder(dtype=tf.float32,shape=[None,224,224,3],name='input')
    y=tf.placehodler(dtype=tf.float32,shape=[None,num_classes],name='label')
    #keep_prob=tf.placehodler(tf.float32)

    #建立模型
    with slim.arg_scope(vgg_arg_scope()):
        outputs,end_points=vgg_16(x,num_classes)

    #定义变量
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs,label=y))
    train_step=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs,1),tf.argmax(y,1)),tf.float32))

    #读取batch
    images, labels = read_and_decode('./train.tfrecords')
    img_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                    batch_size=batch_size,
                                                    capacity=392,
                                                    min_after_dequeue=200)
    label_batch = tf.one_hot(label_batch, num_classes, 1, 0)


    init=tf.global_variables_initializer
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(max_steps):
            batch_x,batch_y=sess.run(img_batch,label_batch)
            _,loss_val=sess.run([train_step,loss],feed_dict={x:batch_x,y:batch_y})

            #每10step 打印
            if i%10==0:
                train_arr=accuracy.eval(feed_dict={x:batch_x,y:batch_y})
                print("%s:step [%d] Loss: %f,training accuracy : %g" %(datetime.now(),i,loss_val,train_arr))

            if (i+1)==max_steps:
                saver.save(sess,'./model/model.ckpt',global_step=i)

        coord.request_stop()
        coord.join(threads)




if __name__=='__main__':
    train()
