#encoding=utf-8
#两层简单神经网络（全连接）
import tensorflow as tf

#定义输入和参数
#正态分布:tf.random_normal([2,3],stddev=2,mean=0,seed=1)) 产生2x3矩阵 标准差为2 均值为0  随机种子
#平均分布:tf.random_uniform()
#去掉过大偏离点的正太分布:tf.truncated_normal()
#全0数组:tf.zeros([3,2],int32) 生成[[0,0],[0,0],[0,0]]
#全1数组tf.ones([3,2],int32) 生成[[1,1],[1,1],[1,1]]
#全定值数组tf.fill([3,2],6) 生成[[6,6],[6,6],[6,6]]
#直接给值tf.constant([3,2,1]) 生成[3,2,1]
x=tf.constant([[0.7,0.5]])
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#定义前向传播过程
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#用会话计算结果
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    print("y in tf_3.py is:\n",sess.run(y))
