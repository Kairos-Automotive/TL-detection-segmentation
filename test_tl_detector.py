import cv2
from matplotlib import pyplot as plt

from tl_detector_segmentation import TLDetectorSegmentation


if __name__ == '__main__':
    detector = TLDetectorSegmentation()
    image = cv2.imread('test_img/carla1.jpg')
    tl_imgs, tf_ms, img_ms = detector.detect(image)
    n = len(tl_imgs)
    if n > 0:
        for i, tl_img in enumerate(tl_imgs):
            ax_i = plt.subplot2grid((1, n), (0, i))
            ax_i.imshow(cv2.cvtColor(tl_img, cv2.COLOR_BGR2RGB))
        plt.show()
    print('tf call {} ms, img process {} ms'.format(tf_ms, img_ms))
