%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import math
import os
import cifar10
from cifar10 import img_size, num_channels, num_classes

img_size_cropped = 24
session = tf.Session()
save_dir = 'checkpoints_alex/'
save_path = os.path.join(save_dir, 'cifar10_cnn')
cifar10.maybe_download_and_extract()
images_train, cls_train, labels_train = cifar10.load_training_data()

def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    
    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image

def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images

def random_batch(images_train, labels_train, size):
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch

def conv(x, out, name, groups, kernel_size=3, stride_x=1, stride_y=1, padding='SAME'):
    with tf.name_scope(name):
        depth = x.get_shape().as_list()[-1] // groups
        convolve = lambda input_, filter_: \
            tf.nn.conv2d(input_, filter_, [1, stride_x, stride_y, 1], padding)
        W = weight_variable(shape=[kernel_size, kernel_size, depth, out], name='W')
        b = bias_variable((out,), name="b")
        if groups == 1:
            conv = convolve(x, W)
        else:
            x_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            w_groups = tf.split(axis=3, num_or_size_splits=groups, value=W)
            conv = tf.concat(axis=3, values=[convolve(i, k) for i, k in zip(x_groups, w_groups)])

        return tf.nn.relu(tf.nn.bias_add(conv, b), name=name)
def dropout(x, keep_prob=0.5):
    return tf.nn.dropout(x, keep_prob)

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.3, shape=shape)
    return tf.Variable(initial, name=name)

def full(x, n_out, name, activ=tf.nn.relu):
    with tf.name_scope(name):
        flattened = tf.reshape(x, [-1, int(np.prod(x.shape[1:]))])
        return tf.layers.dense(flattened, n_out, activation=activ)

def max_pool(x, name, kernel_size=3, stride_x=2, stride_y=2, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                         strides=[1, stride_x, stride_y, 1],
                         padding=padding, name=name)

def lrn(x, *, radius=2, alpha=1e-4, bias=2, beta=0.75, name):
    return tf.nn.lrn(x, depth_radius=radius, bias=bias, alpha=alpha, beta=beta, name=name)
    
class AlexNet:
    def __init__(self):
        self.batch_size = 32
        self.save_path = save_path
        
        self.create_placeholder()
        self.create_model()
        self.create_filds()
     
    def create_placeholder(self):
        self.images = tf.placeholder(shape=[self.batch_size, img_size, img_size, num_channels], dtype=tf.float32, name='img')
        self.x = pre_process(images=self.images, training=True)
        self.y_true = tf.placeholder(shape=[self.batch_size, num_classes], dtype=tf.int64, name='y_true')
        self.y_true_cls = tf.argmax(self.y_true, axis=1)

    def create_model(self):
        # 1
        conv1 = conv(x=self.x, groups=1, kernel_size=7, out=64, name='conv1')
        lrn1 = lrn(conv1, name="lrn1")
        pool1 = max_pool(lrn1, name ='pool2')
        
        # 2
        conv2 = conv(x=lrn1, groups=2, kernel_size=5, out=96, name='conv2')
        pool2 =max_pool(x=conv2, name ='pool2')
        lrn2 = lrn(pool2, name="lrn2")
       
        
        # 3-5
        conv3 = conv(x=lrn2, groups=1, out=192, name='conv3')
        conv4 = conv(conv3, groups=2, out=192, name = 'conv4')
        conv5 = conv(conv4, groups=2, out=128, name = 'conv5')
        pool5 = max_pool(x=conv5, name = 'pool5')

        # 6
        full6 = full(pool5, 128, name='full6')
        dropout6 = dropout(full6)

        # 7
        full7 = full(dropout6, 512, name = 'full7')
        dropout7 = dropout(full7)

        # 8
        full8 = full(dropout7, num_classes, activ=None, name='full8')
        
        self.logits = full8
        print("Here")

    def create_filds(self):
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), axis=1)
        self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true, logits = self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=self.global_step)
        self.saver = tf.train.Saver()
        correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    def load_model(self, session):
        try:
            print("Trying to restore last checkpoint ...")
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=self.save_dir)
            self.saver.restore(session, save_path=last_chk_path)
            print("Restored checkpoint from:", last_chk_path)
        except:
            print("Failed to restore checkpoint. Initializing variablies.")
            session.run(tf.global_variables_initializer())

    def train(self, num_iterations, y_true, images):
        with tf.Session() as session:
            self.load_model(session)
            tf.summary.FileWriter('graphs', session.graph)

            for i in range(num_iterations):
                x_batch, y_true_batch = random_batch(images, y_true, self.batch_size)

                feed_dict_train = {self.images: x_batch, self.y_true: y_true_batch}

                i_global, _ = session.run([self.global_step, self.optimizer],
                                          feed_dict=feed_dict_train)

                if (i_global % 100 == 0) or (i == num_iterations - 1):
                    batch_acc = session.run(self.accuracy,
                                            feed_dict=feed_dict_train)
                    msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
                    print(msg.format(i_global, batch_acc))

                if (i_global % 1000 == 0) or (i == num_iterations - 1):
                    self.saver.save(session,
                               save_path=self.save_path,
                               global_step=self.global_step)

                    print("Saved checkpoint.") 
    def test(self, images, labels, cls_true):
        num_images = len(images)
        cls_pred = np.zeros(shape=num_images, dtype=np.int)

        for i in range(0, n_images, self.batch_size):
            j = min(i + self.batch_size, num_images)
            feed_dict = {self.images: images[i:j, :], self.y_true: labels[i:j, :]}
            cls_pred[i:j] = session.run(self.y_pred_cls, feed_dict)
        correct = (cls_true == cls_pred)
        return cls_pred, correct
        
alex = AlexNet()
alex.train(100000, labels_train, images_train)