import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from tensorflow.python.framework import ops

ops.reset_default_graph()

# 加载iris数据集，存储花瓣的长度作为目标值
iris = datasets.load_iris()
# print(iris.data)
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])

sess = tf.Session()
# 使得返回结果可以复现
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

# 划分训练集和测试集80%和20%，下面的方法是应该学会的。
train_indices = np.random.choice(len(x_vals), round(0.8 * len(x_vals)), replace=False)
# print(train_indices)
# print(list(set(range(len(x_vals)))-set(train_indices)))
# range(len(x_vals))=(0,150) set(range(len(x_vals))是0到149的集合 ，而集合能相互加减，
# 在前面加上list就转换成列表,在前面在加上np.array()转换成数组。
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
# print(test_indices)

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


# 下面是做归一化
def normalize_cols(m):
    col_max = m.max(axis=0)  # 取每一列的最大值
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)


x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))
# nan_to_num()可用来将nan替换成0
# print(y_vals_train)

batch_size = 50
x_data = tf.placeholder(tf.float32, shape=[None, 3])
y_target = tf.placeholder(tf.float32, shape=[None, 1])

hidden_layer_nodes = 10  # 设置五个隐藏节点
W1 = tf.Variable(tf.random_normal(shape=[3, hidden_layer_nodes]))  # W1.shape = (3, 5)
b1 = tf.Variable(tf.random_normal(shape=[1, hidden_layer_nodes]))  # b1.shape = (1, 5)
W2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))  # W2.shape = (5, 1)
b2 = tf.Variable(tf.random_normal(shape=[1]))
# x_data*W1 + b1 (None, 5)
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, W1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, W2), b2))
# 定义损失函数
loss = tf.reduce_mean(tf.square(final_output - y_target))
# 定义优化算法，初始化所有的变量
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

# 遍历迭代训练模型
loss_train = []  # 初始化list存储训练损失
loss_test = []
for i in range(1000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_train.append(np.sqrt(temp_loss))

    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    loss_test.append(np.sqrt(test_temp_loss))
    if (i + 1) % 10 == 0:
        print('Generation: ' + str(i + 1) + '. Loss= ' + str(temp_loss))


# Plot loss (MSE) over time
plt.plot(loss_train, 'k-', label='Train Loss')
plt.plot(loss_test, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
