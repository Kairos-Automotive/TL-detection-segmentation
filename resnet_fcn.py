''' using ResNet-FCN for semantic segmentation
'''
import tensorflow as tf
import math
from tqdm import tqdm
import os

import resnet_fcn_import

class RESNET_FCN:
    def __init__(self, num_classes = 0, define_graph=True, batch_size=None, image_shape=None):
        """ initialize network

        for training we need to set shapes

        for inference just set define_graph=False and call load_model straight away. then use predict only
        """
        self._tag = 'RESNET_FCN'
        self._num_classes = num_classes
        if define_graph:
            # create entire model graph
            self._learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
            self._create_input_pipeline(image_shape, batch_size)
            # this is different
            self._logits = resnet_fcn_import.inference(x=self._images_std, is_training=True, num_classes=num_classes)
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
        #self._keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self._prediction_class = graph.get_tensor_by_name("predictions/prediction_class:0")


    def restore_variables(self, sess, var_values):
        pass


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
            checkpoint_dir = os.path.join(ckpt_dir, self._tag)
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
                                       {self._images: [image]},
                                       options=options,
                                       run_metadata=run_metadata)
        else:
            run_metadata = None
            predicted_class = sess.run( [self._prediction_class], {self._images: [image]})
        predicted_class = predicted_class[0]
        predicted_class = predicted_class[0,:,:,:]
        if trace:
            return predicted_class, run_metadata
        else:
            return predicted_class

    def _create_input_pipeline(self, image_shape, batch_size):
        # define input placeholders in the graph
        with tf.name_scope("data"):
            self._images = tf.placeholder(tf.uint8, name='images', shape=(batch_size, image_shape[0], image_shape[1], 3))
            tf.summary.image('input_images', self._images, max_outputs=2)
            self._labels = tf.placeholder(tf.uint8, name='labels', shape=(batch_size, image_shape[0], image_shape[1], self._num_classes))
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            self._images_float = tf.image.convert_image_dtype(self._images, tf.float32)
            #self._images_std = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self._images_float)
            images_max = tf.add(tf.constant(1.0, dtype=tf.float32), tf.reduce_max(self._images_float, axis=[1,2,3], keep_dims=True))
            images_mean = tf.reduce_mean(self._images_float, axis=[1,2,3], keep_dims=True)
            self._images_std = tf.divide(tf.subtract(self._images_float, images_mean), images_max)
            self._labels_float = tf.cast(self._labels, tf.float32)


    def _create_predictions(self):
        """ define prediction probabilities and classes """
        with tf.name_scope("predictions"):
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
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate, epsilon=1e-8).minimize(self._loss, global_step=self._global_step)


