import tensorflow as tf
import math
from tqdm import tqdm
import os


class FCN8_VGG16:
    def __init__(self, num_classes = 0, define_graph=True):
        """ initialize network

        for training we need to set shapes

        for inference just set define_graph=False and call load_model straight away. then use predict only
        """
        self._tag = 'FCN8'
        self._num_classes = num_classes
        if define_graph:
            # create entire model graph
            self._keep_prob = tf.placeholder(tf.float32, name='keep_prob', shape=[])
            self._learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
            self._create_input_pipeline()
            with tf.name_scope("encoder_vgg16"):
                self._create_vgg16_conv_layers()
                self._create_vgg16_fc_conv_layers()
            self._create_decoder()
            self._create_predictions()
            self._create_optimizer()
            self._summaries = tf.summary.merge_all()
        else:
            # the graph will be loaded from saved model files
            # next call must be with a session object to load_model
            pass

    def load_model(self, sess, model_dir):
        """ load trained model using SavedModelBuilder. can only be used for inference """
        # tf.reset_default_graph()
        # sess.run(tf.global_variables_initializer())
        tf.saved_model.loader.load(sess, [self._tag], model_dir)
        # we need to re-assign the following ops to instance variables for prediction
        # we cannot continue training from this state as other instance variables are undefined
        graph = tf.get_default_graph()
        self._images = graph.get_tensor_by_name("data/images:0")
        self._keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self._prediction_class = graph.get_tensor_by_name("predictions/prediction_class:0")

    def restore_variables(self, sess, var_values):
        # restore trained weights for VGG
        for var in self._parameters:
            name = var.name.replace('encoder_vgg16/', '').replace(':0','')
            value = var_values[name]
            if name=='conv6/weights':
                # this is weird -- Udacity provided model has weights shape of (7,7,512,4096)
                # but it should be (1,1,512,4096). lets take just one filter
                value = value[4:5,4:5,:,:]
            sess.run(var.assign(value))

    def save_model(self, sess, model_dir):
        builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
        builder.add_meta_graph_and_variables(sess, [self._tag])
        builder.save()

    def restore_checkpoint(self, sess, checkpoint_dir):
        """ load saved checkpoint. can be used to continue training model across session """
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)

    def train(self, sess,
              epochs, batch_size, get_batches_fn, n_samples,
              keep_prob_value, learning_rate,
              ckpt_dir=None, summaries_dir=None):
        """
        Train neural network and print out the loss during training.
        :param sess: TF Session
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
        """
        # restore from checkpoint if needed
        if ckpt_dir is None:
            saver = None
        else:
            saver = tf.train.Saver()  # by default saves all variables
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            checkpoint_dir = os.path.join(ckpt_dir, 'fcn8vgg16')
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("restored from checkpoint {}".format(ckpt.model_checkpoint_path))

        if summaries_dir is not None:
            summary_writer = tf.summary.FileWriter(summaries_dir, graph=sess.graph)

        step = self._global_step.eval(session=sess)
        if step > 0:
            print("continuing training after {} steps done previously".format(step))

        l = 0.
        for epoch in range(epochs):
            # running optimization in batches of training set
            n_batches = int(math.ceil(float(n_samples) / batch_size))
            batches_pbar = tqdm(get_batches_fn(batch_size),
                                desc='Train Epoch {:>2}/{} (loss _.___)'.format(epoch + 1, epochs),
                                unit='batches',
                                total=n_batches)
            n = 0.
            l = 0.
            for images, labels in batches_pbar:
                feed_dict = {self._images: images,
                             self._labels: labels,
                             self._keep_prob: keep_prob_value,
                             self._learning_rate: learning_rate}
                _, loss, summaries, _, _ = sess.run([self._optimizer,
                                                     self._loss,
                                                     self._summaries,
                                                     self._prediction_class_idx,
                                                     self._batch_mean_iou],
                                              feed_dict=feed_dict)
                n += len(images)
                l += loss * len(images)
                batches_pbar.set_description(
                    'Train Epoch {:>2}/{} (loss {:.3f})'.format(epoch + 1, epochs, l / n))
                # write training summaries for tensorboard every so often
                step = self._global_step.eval(session=sess)
                if step % 10 == 0 and summaries_dir is not None:
                    summary_writer.add_summary(summaries, global_step=step)
                # if i % 100 == 99:  # Record execution stats
                #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #     run_metadata = tf.RunMetadata()
                #     summary, _ = sess.run([merged, train_step],
                #                           feed_dict=feed_dict(True),
                #                           options=run_options,
                #                           run_metadata=run_metadata)
                #     train_writer.add_run_metadata(run_metadata, 'step%d' % i)
                #     train_writer.add_summary(summary, i)
                #     print('Adding run metadata for', i)
                # else:  # Record a summary
                #     summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                #     train_writer.add_summary(summary, i)

            l /= n_samples

            if saver is not None:
                save_path = saver.save(sess, checkpoint_dir, global_step=self._global_step)
                # print("checkpoint saved to {}".format(save_path))
        return l

    def predict_one(self, sess, image, trace=False):
        """
        Generate prediction for one image
        :param sess: TF session
        :param image: scipy image
        :return: predicted classes for all pixels in the image
        """
        if trace:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            predicted_class = sess.run([self._prediction_class],
                                       {self._keep_prob: 1.0, self._images: [image]},
                                       options=options,
                                       run_metadata=run_metadata)
        else:
            run_metadata = None
            predicted_class = sess.run( [self._prediction_class], {self._keep_prob: 1.0, self._images: [image]})
        predicted_class = predicted_class[0]
        predicted_class = predicted_class[0,:,:,:]
        if trace:
            return predicted_class, run_metadata
        else:
            return predicted_class

    def _create_input_pipeline(self):
        # define input placeholders in the graph
        with tf.name_scope("data"):
            self._images = tf.placeholder(tf.uint8, name='images', shape=(None, None, None, 3))
            tf.summary.image('input_images', self._images, max_outputs=2)
            self._labels = tf.placeholder(tf.uint8, name='labels', shape=(None, None, None, self._num_classes))
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            self._images_float = tf.image.convert_image_dtype(self._images, tf.float32)
            self._images_std = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self._images_float)
            self._labels_float = tf.cast(self._labels, tf.float32)

    def _create_vgg16_conv_layers(self):
        self._parameters = []

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(self._images_std, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out, name=scope)
            #tf.summary.histogram('activations', conv1_1)
            self._parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope)
            #tf.summary.histogram('activations', conv1_1)
            self._parameters += [kernel, biases]

        # pool1
        pool1 = tf.nn.max_pool(conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out, name=scope)
            #tf.summary.histogram('activations', conv1_1)
            self._parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope)
            #tf.summary.histogram('activations', conv1_1)
            self._parameters += [kernel, biases]

        # pool2
        pool2 = tf.nn.max_pool(conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out, name=scope)
            #tf.summary.histogram('activations', conv1_1)
            self._parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out, name=scope)
            #tf.summary.histogram('activations', conv1_1)
            self._parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            tf.summary.histogram("weights", kernel)
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out, name=scope)
            #tf.summary.histogram('activations', conv1_1)
            self._parameters += [kernel, biases]

        # pool3
        self._pool3 = tf.nn.max_pool(conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(self._pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out, name=scope)
            #tf.summary.histogram('activations', conv1_1)
            self._parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out, name=scope)
            #tf.summary.histogram('activations', conv1_1)
            self._parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out, name=scope)
            #tf.summary.histogram('activations', conv1_1)
            self._parameters += [kernel, biases]

        # pool4
        self._pool4 = tf.nn.max_pool(conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(self._pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out, name=scope)
            #tf.summary.histogram('activations', conv1_1)
            self._parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out, name=scope)
            #tf.summary.histogram('activations', conv1_1)
            self._parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out, name=scope)
            #tf.summary.histogram('activations', conv1_1)
            self._parameters += [kernel, biases]

        # pool5
        self._pool5 = tf.nn.max_pool(conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool5')

    def _create_vgg16_fc_conv_layers(self):
        # here we create first two FC layers of VGG16, but as 1x1 convolutions

        # fc1 -> conv6
        with tf.name_scope('conv6') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(self._pool5, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(out, name='relu')
            conv6 = tf.nn.dropout(relu, self._keep_prob, name=scope)
            #tf.summary.histogram('dropout', conv6)
            self._parameters += [kernel, biases]

        # fc2 -> conv7
        with tf.name_scope('conv7') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 1, 4096, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
            tf.summary.histogram("weights", kernel)
            conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
            tf.summary.histogram("biases", biases)
            out = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(out, name='relu')
            self._conv7 = tf.nn.dropout(relu, self._keep_prob, name=scope)
            #tf.summary.histogram('dropout', conv7)
            self._parameters += [kernel, biases]

    def _create_decoder(self):
        # with tf.name_scope("decoder"):
        #     with tf.name_scope("1x1"):
        #         kernel = tf.Variable(tf.truncated_normal([1, 1, 4096, num_classes], dtype=tf.float32, stddev=1e-1), name='weights')
        #         tf.summary.histogram("weights", kernel)
        #         conv_1x1 = tf.nn.conv2d(self._conv7, kernel, [1, 1, 1, 1], padding='SAME', name='conv_1x1')
        #     with tf.name_scope("up4"):
        #         channels = 512
        #         kernel = tf.Variable(tf.truncated_normal([4, 4, num_classes, channels], dtype=tf.float32, stddev=1e-1), name='weights')
        #         tf.summary.histogram("weights", kernel)
        #         out_shape = [s for s in conv_1x1.get_shape()]
        #         out_shape[1] *= 2
        #         out_shape[2] *= 2
        #         #out_shape[3] = channels
        #         up4 = tf.nn.conv2d_transpose(conv_1x1, kernel, output_shape=out_shape, strides=[1, 2, 2, 1], padding='SAME', name='up4')
        #         skip4 = tf.add(up4, self._pool4, name='skip4')
        #     with tf.name_scope("up4"):
        #         channels = 256
        #         kernel = tf.Variable(tf.truncated_normal([4, 4, 512, channels], dtype=tf.float32, stddev=1e-1), name='weights')
        #         tf.summary.histogram("weights", kernel)
        #         out_shape = [s for s in skip4.get_shape()]
        #         out_shape[1] *= 2
        #         out_shape[2] *= 2
        #         out_shape[3] = channels
        #         up3 = tf.nn.conv2d_transpose(skip4, kernel, output_shape=out_shape, strides=[1, 2, 2, 1], padding='SAME', name='up3')
        #         skip3 = tf.add(up3, self._pool3, name='skip3')
        #     with tf.name_scope("output"):
        #         channels = num_classes
        #         kernel = tf.Variable(tf.truncated_normal([16, 16, 256, channels], dtype=tf.float32, stddev=1e-1), name='weights')
        #         tf.summary.histogram("weights", kernel)
        #         out_shape = [s for s in skip3.get_shape()]
        #         out_shape[1] *= 8
        #         out_shape[2] *= 8
        #         out_shape[3] = channels
        #         self._output = tf.nn.conv2d_transpose(skip3, kernel, output_shape=out_shape, strides=[1,8,8,1], padding='SAME', name='output')
        with tf.name_scope("decoder"):
            conv_1x1 = tf.layers.conv2d(self._conv7, self._num_classes, kernel_size=1,
                                        strides=(1, 1), padding='SAME',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        name='conv_1x1')
            # in the paper 'initialise to bilinear interpolation'. here we do random initialization
            up4 = tf.layers.conv2d_transpose(conv_1x1, 512,
                                             kernel_size=4, strides=2, padding='SAME',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             name='up4')
            skip4 = tf.add(up4, self._pool4, name='skip4')
            up3 = tf.layers.conv2d_transpose(skip4, 256,
                                             kernel_size=4, strides=2, padding='SAME',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             name='up3')
            skip3 = tf.add(up3, self._pool3, name='skip3')
            self._output = tf.layers.conv2d_transpose(skip3, self._num_classes,
                                             kernel_size=16, strides=8, padding='SAME',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             name='output')

    def _create_predictions(self):
        """ define prediction probabilities and classes """
        with tf.name_scope("predictions"):
            self._logits = tf.identity(self._output, name='logits')
            self._prediction_softmax = tf.nn.softmax(self._logits, name="prediction_softmax")
            self._prediction_class = tf.cast(tf.greater(self._prediction_softmax, 0.5), dtype=tf.float32, name='prediction_class')
            self._prediction_class_idx = tf.cast(tf.argmax(self._prediction_class, axis=3), dtype=tf.uint8, name='prediction_class_idx')
            tf.summary.image('prediction_class_idx',
                             tf.expand_dims(
                                 tf.cast(self._prediction_class_idx, dtype=tf.float32),
                                 -1),
                             max_outputs=2)
        with tf.name_scope("iou"):
            mul = tf.multiply(self._prediction_class, self._labels_float)
            inter = tf.reduce_sum(mul, axis=[1,2], name='intersection')
            add = tf.add(self._prediction_class, self._labels_float)
            union = tf.add(tf.cast(tf.count_nonzero(tf.subtract(add, mul), axis=[1,2]), dtype=tf.float32), 1e-6, name='union')
            self._iou = tf.divide(inter, union, name='iou')
            tf.summary.histogram("iou", self._iou)
            self._mean_iou = tf.reduce_mean(self._iou, axis=[1], name='mean_iou')
            tf.summary.histogram("mean_iou", self._mean_iou)
            self._batch_mean_iou = tf.reduce_mean(self._mean_iou, name='batch_mean_iou')
            tf.summary.scalar("batch_mean_iou", self._batch_mean_iou)

    def _create_optimizer(self):
        # TODO: use weighted loss to re-balance the classes
        # Can use this https://blog.fineighbor.com/tensorflow-dealing-with-imbalanced-data-eb0108b10701
        # But it is hard problem to solve with TF at this stage. Could not make tf.gather to work.
        # Could feed class weight in same way as labels. But do not have time to do this
        # self._label_weights = tf.tile(tf.reduce_sum(self._labels_float, axis=[1,2], keep_dims=True), [1,tf.shape(self._logits)[1],tf.shape(self._logits)[2], 1])
        # self._label_weights = tf.reduce_sum(self._labels_float, axis=[1,2])
        # tf.summary.histogram("label_weights", self._label_weights)
        # self._loss = tf.losses.sparse_softmax_cross_entropy(labels=self._prediction_class_idx,
        #                                                     logits=self._logits,
        #                                                     weights=self._label_weights)
        self._loss = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(logits=self._logits, labels=self._labels_float),
                                name="loss")
        tf.summary.scalar('loss', self._loss)
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        tf.summary.scalar('global_step', self._global_step)
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss, global_step=self._global_step)


