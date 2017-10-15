from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
import cv2
from scipy.ndimage.measurements import label


class TLDetectorSegmentation:
    def __init__(self):
        # session
        self._session = self._create_session()
        # graph
        self._detector_graph_file = 'detector_graph.pb'
        self._detector_graph_scope = 'detector'
        self._load_detector_graph()
        graph = self._session.graph
        # inputs/outputs
        self._detector_input = graph.get_tensor_by_name(self._detector_graph_scope+'/data/images:0')
        self._detector_keep_prob = graph.get_tensor_by_name(self._detector_graph_scope+'/keep_prob:0')
        self._detector_output = graph.get_tensor_by_name(self._detector_graph_scope+'/predictions/prediction_class:0')
        # image size
        self._image_shape = (288, 384)
        # run on fake image once
        fake_img = np.zeros(self._image_shape+(3,), dtype=np.uint8)
        self.detect(fake_img)


    def _get_labeled_bboxes(self, heatmap):
        # Use labels() from scipy.ndimage.measurements
        # to group pixel blobs into instances of traffic lights
        labels = label(heatmap)
        bboxes = []
        for tl_number in range(1, labels[1] + 1):
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
            if w < 4 or h < 8:
                continue
            bboxes.append(bbox)
        return bboxes


    def detect(self, img):
        """
        Detect bounding boxes for traffic lights in the image

        :param img: arbitrary size image (hopefully aspect ratio close to 4:3) in OpenCV BGR uint8 format
        :return: (list of images of detected traffic lights (in OpenCV format), time of tf run, time of image extraction)
        """
        h, w = img.shape[0], img.shape[1]
        h_sized, w_sized = self._image_shape[0], self._image_shape[1]
        resized_image = cv2.resize(img, (w_sized, h_sized,), interpolation=cv2.INTER_NEAREST)

        # run TF prediction
        start_time = timer()
        predicted_class = self._session.run( [self._detector_output],
                                             {self._detector_keep_prob: 1.0,
                                              self._detector_input: [resized_image]})
        predicted_class = np.array(predicted_class[0][0,:,:,:], dtype=np.uint8)
        duration = timer() - start_time
        tf_time_ms = int(duration * 1000)

        # translate to traffic light images
        start_time = timer()
        label = 1  # only take 'traffic light' class pixels. 0 is background
        segmentation = np.expand_dims(predicted_class[:, :, label], axis=2)
        # calculate bounding boxes
        bboxes = self._get_labeled_bboxes(segmentation)
        # extract bounding boxes on segmented image
        tl_imgs = []
        for i, box in enumerate(bboxes):
            bx1 = box[0][0]
            by1 = box[0][1]
            bx2 = box[1][0]
            by2 = box[1][1]
            # cast back to original image coordinates
            x1 = int(bx1 * w / w_sized)
            x2 = int(bx2 * w / w_sized)
            y1 = int(by1 * h / h_sized)
            y2 = int(by2 * h / h_sized)
            tl_image = img[y1:y2, x1:x2]
            tl_imgs.append(tl_image)
        duration = timer() - start_time
        img_time_ms = int(duration * 1000)

        return tl_imgs, tf_time_ms, img_time_ms


    def _create_session(self):
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        return tf.Session(config=config)

    def _load_detector_graph(self):
        graph_def = tf.GraphDef()
        with open(self._detector_graph_file, 'rb') as f:
            serialized = f.read()
            graph_def.ParseFromString(serialized)
        tf.import_graph_def(graph_def, name=self._detector_graph_scope)
