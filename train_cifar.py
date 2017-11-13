import os
import pickle
import numpy as np
import tensorflow as tf
import math
import skimage as ski
import skimage.io
import tensorflow.contrib.layers as layers
import tensorflow.contrib.metrics as metrics
import matplotlib.pyplot as plt
import time


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[3]
    num_channels = w.shape[2]
    k = w.shape[0]
    assert w.shape[0] == w.shape[1]
    w = w.reshape(k, k, num_channels, num_filters)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k, c:c+k, :] = w[:, :, :, i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def draw_image(img, mean, std):
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()


def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['valid_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    # ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
    #          linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    # ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
    #          linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    # ax3.plot(x_data, data['lr'], marker='o', color=train_color,
    #          linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.pdf')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)


def build_model(num_classes):
    weight_decay = 1e-3
    conv1sz = 16
    conv2sz = 32
    fc3sz = 256
    fc4sz = 128

    inputs = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels = tf.placeholder(tf.int32, [None])

    one_hot = tf.one_hot(labels, num_classes)

    with tf.contrib.framework.arg_scope(
        [layers.convolution2d],
        kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
        weights_initializer=layers.variance_scaling_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)
    ):
        net = layers.convolution2d(inputs, conv1sz, scope='conv1')
        # ostatak konvolucijskih i pooling slojeva
        pool1 = layers.max_pool2d(inputs=net, kernel_size=3, stride=2)
        conv2 = layers.convolution2d(pool1, conv2sz, scope="conv2")
        pool2 = layers.max_pool2d(inputs=conv2, kernel_size=3, stride=2)

    with tf.contrib.framework.arg_scope(
        [layers.fully_connected],
        activation_fn=tf.nn.relu,
        weights_initializer=layers.variance_scaling_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)
    ):
        # sada definiramo potpuno povezane slojeve
        # ali najprije prebacimo 4D tenzor u matricu
        net = layers.flatten(pool2)  # originally inputs...?
        net = layers.fully_connected(net, fc3sz, scope='fc3')
        net = layers.fully_connected(net, fc4sz, scope='fc4')

    logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hot, logits))
    accuracy = metrics.accuracy(tf.argmax(logits, 1), tf.argmax(one_hot, 1))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    return logits, loss, train_op, accuracy, inputs, labels


def evaluate(loss, accuracy, train_x, train_y):
    return sess.run([loss, accuracy], feed_dict={node_x: train_x, node_y: train_y})


DATA_DIR = '/home/koki/faks/dubuce/deep-learning-lab2/CIFAR-10/cifar-10-batches-py'
SAVE_DIR = '/home/koki/faks/dubuce/deep-learning-lab2/SAVE'
img_height = 32
img_width = 32
num_channels = 3
num_epochs = 8
batch_size = 50
total_size = 50000
valid_size = 5000
train_size = total_size - valid_size
num_batches = int(train_size / batch_size)


train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
    subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size]
valid_y = train_y[:valid_size]
train_x = train_x[valid_size:]
train_y = train_y[valid_size:]
data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

logits, loss, train_op, accuracy, node_x, node_y = build_model(10)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

conv1_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1')[0]
conv1_weights = sess.run(conv1_var)
draw_conv_filters(0, 0, conv1_weights, SAVE_DIR)

plot_data = {}
plot_data['train_loss'] = []
plot_data['valid_loss'] = []
plot_data['train_acc'] = []
plot_data['valid_acc'] = []
plot_data['lr'] = []
for epoch_num in range(1, num_epochs + 1):
    train_x, train_y = shuffle_data(train_x, train_y)
    for step in range(num_batches):
        offset = step * batch_size
        # s ovim kodom pazite da je broj primjera djeljiv s batch_size
        batch_x = train_x[offset:(offset + batch_size)]
        batch_y = train_y[offset:(offset + batch_size)]
        feed_dict = {node_x: batch_x, node_y: batch_y}
        start_time = time.time()
        run_ops = [train_op, loss, logits]
        ret_val = sess.run(run_ops, feed_dict=feed_dict)
        _, loss_val, logits_val = ret_val
        duration = time.time() - start_time
        if (step+1) % 50 == 0:
            sec_per_batch = float(duration)
            format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
            print(format_str % (epoch_num, step+1, num_batches, loss_val, sec_per_batch))

    # print('Train error:')
    # train_loss, train_acc = evaluate(loss, accuracy, train_x, train_y)
    print('Validation error:')
    valid_loss, valid_acc = evaluate(loss, accuracy, valid_x, valid_y)
    print(valid_loss)
    print(valid_acc)
    # plot_data['train_loss'] += [train_loss]
    plot_data['valid_loss'] += [valid_loss]
    # plot_data['train_acc'] += [train_acc]
    plot_data['valid_acc'] += [valid_acc]
    # plot_data['lr'] += [lr.eval(session=sess)]
    plot_training_progress(SAVE_DIR, plot_data)
conv1_weights = sess.run(conv1_var)
draw_conv_filters(num_epochs, num_batches, conv1_weights, SAVE_DIR)
