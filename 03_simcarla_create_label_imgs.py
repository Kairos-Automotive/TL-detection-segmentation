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
    """
    - boxes:
      - {label: Yellow, occluded: false, x_max: 527.1012443072, x_min: 518.7379907959,
        y_max: 296.3838188664, y_min: 278.7432988131}
      path: ./rgb/additional/2015-10-05-10-55-33_bag/56988.png
    
    - annotations:
      - {class: Green, x_width: 52.65248226950354, xmin: 130.4964539007092, y_height: 119.60283687943263,
        ymin: 289.36170212765956}
      - {class: Green, x_width: 50.156028368794296, xmin: 375.60283687943263, y_height: 121.87234042553195,
        ymin: 293.90070921985813}
      - {class: Green, x_width: 53.33333333333326, xmin: 623.6595744680851, y_height: 119.82978723404256,
        ymin: 297.7588652482269}
      class: image
      filename: sim_data_capture/left0003.jpg
    """
    for i in range(len(images)):
        images[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml), images[i]['filename']))
        images[i]['boxes'] = []
        for ann in images[i]['annotations']:
            box = {}
            box['x_min'] = ann['xmin']
            box['y_min'] = ann['ymin']
            box['x_max'] = ann['xmin'] + ann['x_width']
            box['y_max'] = ann['ymin'] + ann['y_height']
            images[i]['boxes'].append(box)
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
            fname = image_dict['path'].replace('.jpg','_labels.png')
            scipy.misc.imsave(fname, label_image)
        j += 1


if __name__ == '__main__':
    label_file = 'data/sim/vatzal/sim_data_annotations.yaml'
    create_label_images(label_file)
    label_file = 'data/carla/vatzal/real_training_data/real_data_annotations.yaml'
    create_label_images(label_file)

