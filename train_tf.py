import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data


def build_model(inputs, labels, num_classes):
    weight_decay = 1e-3
    conv1sz = 16
    conv2sz = 32
    fc3sz = 512
    with tf.contrib.framework.arg_scope(
        [layers.convolution2d],
        kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
        weights_initializer=layers.variance_scaling_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)
    ):
        net = layers.convolution2d(inputs, conv1sz, scope='conv1')
        # ostatak konvolucijskih i pooling slojeva
        pool1 = layers.max_pool2d(inputs=net, kernel_size=[2, 2], stride=2)
        conv2 = layers.convolution2d(pool1, conv2sz, scope="conv2")
        pool2 = layers.max_pool2d(inputs=conv2, kernel_size=2, stride=2)

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

    logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(labels, logits))

    return logits, loss


DATA_DIR = '/home/koki/faks/dubuce/deep-learning-lab2/DATA'
SAVE_DIR = "/home/koki/faks/dubuce/deep-learning-lab2/SAVE"

dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)

valid_x = dataset.validation.images
valid_x = valid_x.reshape([-1, 28, 28, 1])
valid_y = dataset.validation.labels
test_x = dataset.test.images
test_x = test_x.reshape([-1, 28, 28, 1])
test_y = dataset.test.labels

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

logits, loss = build_model(x, y, 10)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

session = tf.Session()

init = tf.global_variables_initializer()
session.run(init)
# optimizacijska petlja

for epoch in range(8):
    for batch_num in range(1100):
        batch = dataset.train.next_batch(50)
        l, _ = session.run([loss, train_op], feed_dict={
            x: batch[0].reshape([-1, 28, 28, 1]),
            y: batch[1].reshape([-1, 10])})
        if batch_num % 5 == 0:
            print("epoch " + str(epoch) + ", step " + str((batch_num*50)) + "/55000, batch loss = " + str(l))
    l = session.run(loss, feed_dict={
        x: valid_x,
        y: valid_y
    })
    print("Validation loss: " + str(l))
l = session.run(loss, feed_dict={
    x: test_x,
    y: test_y
})
print("Test loss: " + str(l))
