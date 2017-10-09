#!/usr/bin/env python
"""
Create png 'label' images corresponding to street scene pictures. Similar to Cityscapes approach
"""
import sys
import os
#import cv2

import yaml
import scipy.misc
import numpy as np


def get_all_labels(input_yaml, riib=False):
    """ Gets all labels within label file

    Note that RGB images are 1280x720 and RIIB images are 1280x736.
    :param input_yaml: Path to yaml file
    :param riib: If True, change path to labeled pictures
    :return: images: Labels for traffic lights
    """
    images = yaml.load(open(input_yaml, 'rb').read())

    for i in range(len(images)):
        images[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml), images[i]['path']))
        if riib:
            images[i]['path'] = images[i]['path'].replace('.png', '.pgm')
            images[i]['path'] = images[i]['path'].replace('rgb/train', 'riib/train')
            images[i]['path'] = images[i]['path'].replace('rgb/test', 'riib/test')
            for box in images[i]['boxes']:
                box['y_max'] = box['y_max'] + 8
                box['y_min'] = box['y_min'] + 8
    return images


def ir(some_value):
    """Int-round function for short array indexing """
    return int(round(some_value))

def createLabelImage(annotation, encoding):
    size = ( annotation.imgWidth , annotation.imgHeight )
    background = 0
    labelImg = Image.new("L", size, background)
    # a drawer to draw into the image
    drawer = ImageDraw.Draw( labelImg )

    # loop over all objects
    for obj in annotation.objects:
        polygon = obj.polygon
        val = 1
        drawer.polygon( polygon, fill=val )

    return labelImg


def create_label_images(input_yaml):
    images = get_all_labels(input_yaml)

    j = 1
    for i, image_dict in enumerate(images):
        print("processing image {}: {}".format(j, image_dict['path']))
        image = scipy.misc.imread(image_dict['path'])
        shape = image.shape
        if image is None:
            raise IOError('Could not open image path', image_dict['path'])
        label_image = np.zeros((shape[0], shape[1],), dtype=np.uint8)
        is_good = False
        for box in image_dict['boxes']:
            xmin = ir(box['x_min'])
            ymin = ir(box['y_min'])
            xmax = ir(box['x_max'])
            ymax = ir(box['y_max'])
            # exclude invalid boxes
            if xmax-xmin<=0 or ymax-ymin<=0:
                continue
            # exclude too small boxes
            if xmax-xmin<=10 or ymax-ymin<=20:
                continue
            label_image[ymin:(ymax+1), xmin:(xmax+1)] = 1
            is_good = True
        if is_good:
            fname = image_dict['path'].replace('.png','_labels.png')
            scipy.misc.imsave(fname, label_image)
        j += 1


if __name__ == '__main__':
    label_file = 'data/bosch/additional_train.yaml'
    create_label_images(label_file)

    label_file = 'data/bosch/test.yaml'
    create_label_images(label_file)

    label_file = 'data/bosch/train.yaml'
    create_label_images(label_file)

