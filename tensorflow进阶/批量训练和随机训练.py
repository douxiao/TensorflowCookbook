import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

#  生成数据

x_vals = np.random.normal(1., 0.1, 100)
y_vals = np.repeat(10., 100)

x_data = tf.placeholder(tf.float32, shape=[1])
y_target = tf.placeholder(tf.float32, shape=[1])

# 创建一个随机变量模型参数A
A = tf.Variable(tf.random_normal(shape=[1]))
output = tf.multiply(A, x_data)
# 定义一个损失函数L2
loss = tf.square(output - y_target)

# 初始化随机变量
init = tf.global_variables_initializer()
sess.run(init)


# 定义一个优化算法
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)


loss_stochastic = []
# run loop
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 5 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_stochastic.append(temp_loss)

sess.close()

# 批量训练
# 需要重新初始化
ops.reset_default_graph()
sess = tf.Session()

batch_size =20
x_vals = np.random.normal(1., 0.1, 100)
y_vals = np.repeat(10., 100)

x_data = tf.placeholder(tf.float32, shape=[None, 1])
y_target = tf.placeholder(tf.float32, shape=[None, 1])

A = tf.Variable(tf.random_normal([1, 1]))

my_output = tf.matmul(x_data, A)

loss = tf.reduce_mean(tf.square(my_output - y_target))
# 初始化随机变量
init = tf.global_variables_initializer()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)

loss_batch = []

for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 5 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_batch.append(temp_loss)

plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='Stochastic Loss')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch Loss, size =20')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()




































