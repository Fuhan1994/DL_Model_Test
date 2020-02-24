"""
Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim=tf.contrib.slim

def vgg_arg_scope(weight_decay=0.0005):
    """
    define the vgg arg scope   
     
    """
    # 定义默认参数
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d],padding='SAME')as arg_sc:
            return arg_sc

# input=224*224
# conv1_1 64 [3,3]
# conv1_2 64 [3,3]
# maxpool
# conv2_1 128 [3,3]
# conv2_2 128 [3,3]
#
# conv3_1 256 [3,3]
# conv3_2 256 [3,3]
# conv3_3 256 [3,3]
#
# conv4_1 512 [3,3]
# conv4_2 512 [3,3]
# conv4_3 512 [3,3]
#
# conv5_1 512 [3,3]
# conv5_2 512 [3,3]
# conv5_3 512 [3,3]

# flatten1 4096
# flatten2 4096
# flatten3 100
# soft-max

def vgg_16(inputs,num_classes=1000,is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_16',
          fc_conv_padding='VALID'):
    """
    :param inputs: [batch_size,height,width,channels]
    :param num_classes: number of predicted classes
    :param is_training: 
    :param dropout_keep_prob: the probability that activations are kept in the dropout
    :param spatial_squeeze: whether or not should squeeze the spatial dimensions of the outputs.useful
                            to remove unnecessary dimensions for classificaion.
    :param scope: 
    :param fc_conv_padding: the type of padding to use for the fully connected layer that is implemented as
                           a convolutional layer
    :return: the last op containing the log predictions and end_points dict
    """

    # 管理传给get_variable()变量名称的作用域
    with tf.variable_scope(scope,'vgg_16',[inputs]) as sc:
        end_points_collextion=sc.name+'_end_points'
        # collect outputs for conv2d,fully_connected and max_pool2d
        #将conv2d,fully_connected max_pool2d 值存入collection，可以通过tf.get_collextion()获得
        with slim.arg_scope([slim.conv2d,slim.fully_connected,slim.max_pool2d],
                            outputs_collections=end_points_collextion):
            #slim.repeat(inputs,repeat_num,repeat_function,num_outputs,kernel_size,scope)
            #slim.conv2d(inputs,num_outputs,kernel_size,stride=1,padding='SAME'
            #            data_format=None,rate=1,activation_fn=nn.relu,normalizer_fn=None.
            #            normalizer_params=None,weights_initializer=initializer.xavier_initializer(),
            #            biases_initializer=init_ops.zeros_initializer(),
            #            reuse=None,
            #            variablex_collextions=None
            #            outputs=collextions=None,
            #            trainable=True,scope=None)
            #slim.max_pool2d()
            net=slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope='conv1')
            net=slim.max_pool2d(net,[2,2],scope='poool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            #用卷积层替代全连接层， 并行计算，可以加开模型的优化
            net=slim.conv2d(net,4096,[7,7],padding=fc_conv_padding,scope='fc6')
            net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope='dropout6')
            net=slim.conv2d(net,4096,[1,1],scope='fc7')
            net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope='dropout7')
            net=slim.conv2d(net,num_classes,[1,1],activation_fc=None,normalizer=None,scope='fc8')
            #slim.utils.convert_collextion_to_dict() 集合转换字典，
            end_points=slim.utils.convert_collextion_to_dict(end_points_collextion)
            if spatial_squeeze:
                net=tf.squeeze(net,[1,2],name='fc8/squeezed')
                end_points['predictions']=slim.softmax(net,scope='predictions')
                return net,end_points

vgg_16.default_image_size=224




