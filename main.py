import os.path
import os
import warnings
from distutils.version import LooseVersion
import shutil
import time
import argparse
import glob
import random
from timeit import default_timer as timer
import math

import tensorflow as tf
from tensorflow.python.framework import graph_util as tf_graph_util
from tensorflow.python.client import timeline
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import label
import matplotlib.patches as patches

import helper
import fcn8vgg16v2

"""Using FCN Semantic Segmentation to detect Traffic Lights.

Architecture as in https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

Trained on Citiscapes data https://www.cityscapes-dataset.com/
Trained on Bosch Traffic Light data https://hci.iwr.uni-heidelberg.de/node/6132
"""



def load_trained_vgg_vars(sess):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :return: dict name/value where value is pre-trained array
    """
    # Download pretrained vgg model
    vgg_path = 'pretrained_vgg/vgg'
    helper.maybe_download_pretrained_vgg(vgg_path)
    # load model
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    # extract variables
    graph = tf.get_default_graph()
    variables = [op for op in graph.get_operations() if op.op_def and op.op_def.name[:5] == 'Varia']
    # filter out relevant variables and change names
    var_values = {}
    for var in variables:
        name = var.name
        tensor = tf.get_default_graph().get_tensor_by_name(name + ':0')
        value = sess.run(tensor)
        name = name.replace('filter', 'weights')
        name = name.replace('fc', 'conv')
        var_values[name] = value
    return var_values


def get_train_batch_generator(label_path_patterns, image_shape):
    """
    Create batch generator for batches of training data. The label paths are inferred
    :param city_labels_path_pattern: path pattern for Cityscapes images
    :param bosch_labels_path_pattern: path pattern for Bosch Traffic Lights images
    :param image_shape: Tuple - Shape of image
    :return:
    """
    label_paths = []
    image_paths = {}

    if 'city' in label_path_patterns:
        city_labels_path_pattern = label_path_patterns['city']
        city_label_paths = glob.glob(city_labels_path_pattern)
        city_image_paths = {}
        for lb_path in city_label_paths:
            im_path = lb_path.replace('/gtFine/', '/leftImg8bit/')
            im_path = im_path.replace('/gtCoarse/', '/leftImg8bit/')
            im_path = im_path.replace('_gtCoarse_labelTrainIds.png', '_leftImg8bit.png')
            im_path = im_path.replace('_gtFine_labelTrainIds.png', '_leftImg8bit.png')
            if not os.path.exists(im_path):
                raise Exception('cannot find image path corresponding to label {}'.format(lb_path))
            city_image_paths[lb_path] = im_path
        print("Cityscapes training examples: {}".format(len(city_label_paths)))
        label_paths += city_label_paths
        image_paths.update(city_image_paths.copy()) # https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression

    if 'bosch' in label_path_patterns:
        bosch_labels_path_pattern = label_path_patterns['bosch']
        bosch_label_paths = [n for n in glob.glob(bosch_labels_path_pattern) if n.find('_labels.png')>-1]
        bosch_image_paths = {}
        for lb_path in bosch_label_paths:
            im_path = lb_path.replace('_labels.png', '.png')
            if not os.path.exists(im_path):
                raise Exception('cannot find image path corresponding to label {}'.format(lb_path))
            bosch_image_paths[lb_path] = im_path
        print("Bosch training examples: {}".format(len(bosch_label_paths)))
        label_paths += bosch_label_paths
        image_paths.update(bosch_image_paths.copy())

    if 'sim' in label_path_patterns:
        sim_labels_path_pattern = label_path_patterns['sim']
        sim_label_paths = [n for n in glob.glob(sim_labels_path_pattern) if n.find('_labels.png')>-1]
        sim_image_paths = {}
        for lb_path in sim_label_paths:
            im_path = lb_path.replace('_labels.png', '.jpg')
            if not os.path.exists(im_path):
                raise Exception('cannot find image path corresponding to label {}'.format(lb_path))
            sim_image_paths[lb_path] = im_path
        print("Simulator training examples: {}".format(len(sim_label_paths)))
        label_paths += sim_label_paths
        image_paths.update(sim_image_paths.copy())

    if 'carla' in label_path_patterns:
        carla_labels_path_pattern = label_path_patterns['carla']
        carla_label_paths = [n for n in glob.glob(carla_labels_path_pattern) if n.find('_labels.png')>-1]
        carla_image_paths = {}
        for lb_path in carla_label_paths:
            im_path = lb_path.replace('_labels.png', '.jpg')
            if not os.path.exists(im_path):
                raise Exception('cannot find image path corresponding to label {}'.format(lb_path))
            carla_image_paths[lb_path] = im_path
        print("CARLA training examples: {}".format(len(carla_label_paths)))
        label_paths += carla_label_paths
        image_paths.update(carla_image_paths.copy())

    num_classes = 2
    num_samples = len(label_paths)
    assert len(image_paths) == len(label_paths)

    def get_batches_fn(batch_size):
        """
        Create batches of training data.
        Uses OpenCV imread so the format of read images is BGR.
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        random.shuffle(label_paths)
        count = 0
        for batch_i in range(0, num_samples, batch_size):
            images = []
            gt_images = []
            for label_file in label_paths[batch_i:batch_i+batch_size]:
                image_file = image_paths[label_file]
                image = cv2.resize(cv2.imread(image_file), (image_shape[1],image_shape[0],), interpolation=cv2.INTER_NEAREST)
                gt_image = cv2.resize(cv2.imread(label_file), (image_shape[1],image_shape[0],), interpolation=cv2.INTER_NEAREST)
                gt_image = gt_image[:,:,0] # openCV reads grayscale png image as 3 channels
                tmp = []
                for label in range(num_classes):
                  tmp.append(gt_image == label)
                gt_image = np.array(np.stack(tmp, axis=-1), dtype=np.uint8)
                images.append(image)
                gt_images.append(gt_image)
                count += 1
            yield np.array(images), np.array(gt_images)
    return get_batches_fn, num_samples


def train(args, image_shape, labels_path_patterns):
    config = session_config(args)

    # extract pre-trained VGG weights
    with tf.Session(config=config) as sess:
        var_values = load_trained_vgg_vars(sess)
    tf.reset_default_graph()

    with tf.Session(config=config) as sess:
        # define our FCN
        num_classes = 2 # "traffic light"/"not traffic light" pixel
        model = fcn8vgg16v2.FCN8_VGG16(num_classes)

        # variables initialization
        sess.run(tf.global_variables_initializer())
        model.restore_variables(sess, var_values)

        # Create batch generator
        train_batches_fn, num_samples = get_train_batch_generator(label_path_patterns,
                                                                  image_shape)
        print("total number of training examples: {}".format(num_samples))
        time_str = time.strftime("%Y%m%d_%H%M%S")
        run_name = "/{}_ep{}_b{}_lr{:.6f}_kp{}".format(time_str, args.epochs, args.batch_size, args.learning_rate, args.keep_prob)
        start_time = time.time()

        final_loss = model.train(sess, args.epochs, args.batch_size,
                                 train_batches_fn, num_samples,
                                 args.keep_prob, args.learning_rate,
                                 args.ckpt_dir, args.summary_dir+run_name)

        # Make folder for current run
        output_dir = os.path.join(args.runs_dir, time_str)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # save training details to text file
        with open(os.path.join(output_dir, "params.txt"), "w") as f:
            f.write('keep_prob={}\n'.format(args.keep_prob))
            f.write('num_samples={}\n'.format(num_samples))
            f.write('batch_size={}\n'.format(args.batch_size))
            f.write('epochs={}\n'.format(args.epochs))
            f.write('gpu={}\n'.format(args.gpu))
            f.write('gpu_mem={}\n'.format(args.gpu_mem))
            f.write('learning_rate={}\n'.format(args.learning_rate))
            f.write('final_loss={}\n'.format(final_loss))
            duration = time.time() - start_time
            f.write('total_time_hrs={}\n'.format(duration/3600))

        # save model
        """ save trained model using SavedModelBuilder """
        if args.model_dir is None:
            model_dir = os.path.join(output_dir, 'model')
        else:
            model_dir = args.model_dir
        print('saving trained model to {}'.format(model_dir))
        model.save_model(sess, model_dir)


def get_labeled_bboxes(heatmap):
    # Use labels() from scipy.ndimage.measurements to identify 'cars'
    labels = label(heatmap)
    bboxes = []
    for tl_number in range(1, labels[1]+1):
        # Find pixels with each tl_number label value
        nonzero = (labels[0] == tl_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # skip boxes which do not look realistic. too small in one dimension
        w = bbox[1][0] - bbox[0][0]
        h = bbox[1][1] - bbox[0][1]
        if w<5 or h<10:
            continue
        bboxes.append(bbox)
    # Return the bounding boxes
    return bboxes

def predict_image(sess, model, image, trace=False):
    # Adjust image size.
    # Assumes BGR color scheme, as read by OpenCV
    # Input image size is arbitrary and may break middle of decoder in the network.
    # We need to feed FCN images sizes in multiples of 32
    image_shape = [x for x in image.shape]
    fcn_shape = [x for x in image.shape]
    # should be bigger and multiple of 32 for fcn to work
    fcn_shape[0] = int(math.ceil(fcn_shape[0] / 32) * 32)
    fcn_shape[1] = int(math.ceil(fcn_shape[1] / 32) * 32)
    tmp_image = np.zeros(fcn_shape, dtype=np.uint8)
    tmp_image[0:image_shape[0], 0:image_shape[1], :] = image

    # run TF prediction
    start_time = timer()
    if trace:
        predicted_class, run_metadata = model.predict_one(sess, tmp_image, trace=trace)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        with open('tf_trace_timeline.json', 'w') as f:
            f.write(fetched_timeline.generate_chrome_trace_format())
    else:
        predicted_class = model.predict_one(sess, tmp_image)
    predicted_class = np.array(predicted_class, dtype=np.uint8)
    duration = timer() - start_time
    tf_time_ms = int(duration * 1000)

    # adjust predicted_class array back to original sized image
    predicted_class = predicted_class[0:image_shape[0], 0:image_shape[1], :]

    # overlay on image
    start_time = timer()
    label = 1 # only take 'traffic light' class pixels. 0 is background
    segmentation = np.expand_dims(predicted_class[:, :, label], axis=2)
    # calculate bounding boxes
    bboxes = get_labeled_bboxes(segmentation)
    # create segmented image
    color = np.array([[0, 255, 0]], dtype=np.uint8)
    mask = np.dot(segmentation, color)
    transparency_level = 0.5
    segmented_image = np.zeros_like(image)
    cv2.addWeighted(mask, transparency_level, image, 1.0, 0, segmented_image)

    duration = timer() - start_time
    img_time_ms = int(duration * 1000)

    return segmented_image, tf_time_ms, img_time_ms, bboxes


def session_config(args):
    # tensorflow GPU config
    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': args.gpu})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem
    # playing with JIT level, this can be set to ON_1 or ON_2
    if args.xla is not None:
        if args.xla==1:
            jit_level = tf.OptimizerOptions.ON_1 # this works on Ubuntu tf1.3 but does not improve performance
        if args.xla==2:
            jit_level = tf.OptimizerOptions.ON_2
        config.graph_options.optimizer_options.global_jit_level = jit_level


def predict_file(args, image_shape, file_names):
    tf.reset_default_graph()
    with tf.Session(config=session_config(args)) as sess:
        model = fcn8vgg16v2.FCN8_VGG16(define_graph=False)
        model.load_model(sess, 'trained_model' if args.model_dir is None else args.model_dir)

        tf_total_duration = 0.
        img_total_duration = 0.
        tf_count = 0.
        img_count = 0.
        for image_file in file_names:#images_pbar:
            print('Predicting on image {}'.format(image_file))
            image = cv2.resize(cv2.imread(image_file), (image_shape[1],image_shape[0],), interpolation=cv2.INTER_NEAREST)

            # first call for TF to get up to speed
            segmented_image, tf_time_ms, img_time_ms, bboxes = predict_image(sess, model, image)

            # measure real performance
            segmented_image, tf_time_ms, img_time_ms, bboxes = predict_image(sess, model, image, trace=args.trace)

            if tf_count>0:
                tf_total_duration += tf_time_ms
            tf_count += 1
            tf_avg_ms = int(tf_total_duration/(tf_count-1 if tf_count>1 else 1))

            if img_count>0:
                img_total_duration += img_time_ms
            img_count += 1
            img_avg_ms = int(img_total_duration/(img_count-1 if img_count>1 else 1))

            print('tf call {} ms, img process {} ms'.format(tf_time_ms, img_time_ms))

            # tf timings:
            #    mac cpu inference is 540ms on frozen or optimized graph. tf 1.3 from sources
            #    mac cpu inference is 1110ms on frozen or optimized graph. tf 1.3 from pip
            # ubuntu cpu inference is 1100ms on pip tf 1.3
            # ubuntu gpu inference is 12ms on pip tf-gpu 1.3 (cuda 6.0)
            # ubuntu gpu inference is 12ms on custom built tf-gpu 1.3 (cuda+xla).

            # add bounding boxes on segmented image
            n = len(bboxes)
            if n>0:
                ax_im = plt.subplot2grid((2, n), (0, 0), colspan=n)
                ax_im.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
                for i, box in enumerate(bboxes):
                    xy = box[0]
                    w = box[1][0] - xy[0]
                    h = box[1][1] - xy[1]
                    rect = patches.Rectangle(xy, w, h, linewidth=1, edgecolor='r', facecolor='none')
                    ax_im.add_patch(rect)
                    ax_i = plt.subplot2grid((2, n), (1, i))
                    ax_i.imshow(cv2.cvtColor(image[xy[1]:xy[1]+h, xy[0]:xy[0]+w], cv2.COLOR_BGR2RGB))
            else:
                fig, ax_im = plt.subplots()
                ax_im.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
            fig = plt.gcf()
            fig.canvas.set_window_title(image_file)
            plt.draw()
            plt.show()
            #imsave(os.path.join(output_dir, os.path.basename(image_file)), segmented_image)


def freeze_graph(args):
    # based on https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    if args.ckpt_dir is None:
        print("for freezing need --ckpt_dir")
        return
    if args.frozen_model_dir is None:
        print("for freezing need --frozen_model_dir")
        return

    checkpoint = tf.train.get_checkpoint_state(args.ckpt_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    print("freezing from {}".format(input_checkpoint))
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    print("{} ops in the input graph".format(len(input_graph_def.node)))

    output_node_names = "predictions/prediction_class"

    # freeze graph
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # use a built-in TF helper to export variables to constants
        output_graph_def = tf_graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")
        )

    print("{} ops in the frozen graph".format(len(output_graph_def.node)))

    if os.path.exists(args.frozen_model_dir):
        shutil.rmtree(args.frozen_model_dir)

    # save model in same format as usual
    print('saving frozen model as saved_model to {}'.format(args.frozen_model_dir))
    model = fcn8vgg16v2.FCN8_VGG16(define_graph=False)
    tf.reset_default_graph()
    tf.import_graph_def(output_graph_def, name='')
    with tf.Session() as sess:
        model.save_model(sess, args.frozen_model_dir)

    print('saving frozen model as graph.pb (for transforms) to {}'.format(args.frozen_model_dir))
    with tf.gfile.GFile(args.frozen_model_dir+'/graph.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())


def optimise_graph(args):
    """ optimize frozen graph for inference """
    if args.frozen_model_dir is None:
        print("for optimise need --frozen_model_dir")
        return
    if args.optimised_model_dir is None:
        print("for optimise need --optimised_model_dir")
        return

    print('calling c++ implementation of graph transform')
    os.system('./optimise.sh {}'.format(args.frozen_model_dir))

    # reading optimised graph
    tf.reset_default_graph()
    gd = tf.GraphDef()
    output_graph_file = args.frozen_model_dir+"/optimised_graph.pb"
    with tf.gfile.Open(output_graph_file, 'rb') as f:
        gd.ParseFromString(f.read())
    tf.import_graph_def(gd, name='')
    print("{} ops in the optimised graph".format(len(gd.node)))

    # save model in same format as usual
    shutil.rmtree(args.optimised_model_dir, ignore_errors=True)
    #if not os.path.exists(args.optimised_model_dir):
    #    os.makedirs(args.optimised_model_dir)

    print('saving optimised model as saved_model to {}'.format(args.optimised_model_dir))
    model = fcn8vgg16v2.FCN8_VGG16(define_graph=False)
    tf.reset_default_graph()
    tf.import_graph_def(gd, name='')
    with tf.Session() as sess:
        model.save_model(sess, args.optimised_model_dir)
    shutil.move(args.frozen_model_dir+'/optimised_graph.pb', args.optimised_model_dir)


def check_tf():
    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if args.action=='train' and not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    # set tf logging
    tf.logging.set_verbosity(tf.logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',
                        help='what to do: train/train2/predict/freeze/optimise',
                        type=str,
                        choices=['train','train2','predict', 'freeze', 'optimise'])
    parser.add_argument('-g', '--gpu', help='number of GPUs to use. default 0 (use CPU)', type=int, default=0)
    parser.add_argument('-gm','--gpu_mem', help='GPU memory fraction to use. default 0.9', type=float, default=0.9)
    parser.add_argument('-x','--xla', help='XLA JIT level. default None', type=int, default=None, choices=[1,2])
    parser.add_argument('-ep', '--epochs', help='training epochs. default 0', type=int, default=0)
    parser.add_argument('-bs', '--batch_size', help='training batch size. default 5', type=int, default=5)
    parser.add_argument('-lr', '--learning_rate', help='training learning rate. default 0.0001', type=float, default=0.0001)
    parser.add_argument('-kp', '--keep_prob', help='training dropout keep probability. default 0.9', type=float, default=0.9)
    parser.add_argument('-rd', '--runs_dir', help='training runs directory. default runs', type=str, default='runs')
    parser.add_argument('-cd', '--ckpt_dir', help='training checkpoints directory. default ckpt', type=str, default='ckpt')
    parser.add_argument('-sd', '--summary_dir', help='training tensorboard summaries directory. default summaries', type=str, default='summaries')
    parser.add_argument('-md', '--model_dir', help='model directory. default None - model directory is created in runs. needed for predict', type=str, default=None)
    parser.add_argument('-fd', '--frozen_model_dir', help='model directory for frozen graph. for freeze', type=str, default=None)
    parser.add_argument('-od', '--optimised_model_dir', help='model directory for optimised graph. for optimize', type=str, default=None)
    parser.add_argument('-fn', '--file_name', help='file to run prediction on', type=str, default=None)
    parser.add_argument('-tr', '--trace', help='turn tensorflow tracing on for prediction', type=bool, default=False)
    args = parser.parse_args()
    return args




if __name__ == '__main__':

    args = parse_args()

    check_tf()

    print("action={}".format(args.action))
    print("gpu={}".format(args.gpu))
    if args.action=='train' or args.action=='train2':
        print('keep_prob={}'.format(args.keep_prob))
        print('batch_size={}'.format(args.batch_size))
        print('epochs={}'.format(args.epochs))
        print('learning_rate={}'.format(args.learning_rate))
        # this is image size to be read and trained on. predict also uses this
        image_shape = (288, 384)
        if args.action=='train':
            # cityscapes size is 2048x1024, i.e. 2:1.
            # bosch size is 1280x720. i.e. 80x45~1.77.
            label_path_patterns = {
                'city': 'data/cityscapes/*/*/*/*_gt*_labelTrainIds.png',
                'bosch': 'data/bosch/rgb/*/*/*_labels.png'
            }
            train(args, image_shape, label_path_patterns)
        else:
            # sim size is 800x600. 1.33
            # bosch size is 1368x1096. i.e. 1.25
            label_path_patterns = {
                'sim': 'data/sim/vatzal/sim_data_capture/*_labels.png',
                'carla': 'data/carla/vatzal/real_training_data/*/*_labels.png'
            }
            train(args, image_shape, label_path_patterns)

    elif args.action=='predict':
        image_shape = (288, 384)
        print('trace={}'.format(args.trace))
        if args.file_name is not None:
            files = [args.file_name]
        else:
            files = glob.glob('test_img/*.*')
        predict_file(args, image_shape, files)
    elif args.action == 'freeze':
        freeze_graph(args)
    elif args.action == 'optimise':
        optimise_graph(args)
