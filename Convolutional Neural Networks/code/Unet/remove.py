import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from display import create_mask, display
import cv2
import data

def remove(img_path, outline=True):
    """
    이미지의 배경 제거
    """
    model = tf.keras.models.load_model('model.h5')
    my_img = data.read_image(img_path)
    my_img = np.reshape(my_img,(-1,256,256,3))

    pred = model.predict(my_img)
    pred = create_mask(pred)
    pred = np.reshape(pred,(256,256))

    if outline:
      pred_mask = np.where(pred == 1, 0, 1)
    else:
      pred_mask = np.where(pred == 0, 1, 0)

    img = my_img[0]

    rc, gc, bc = cv2.split(img)
    ac = np.ones(rc.shape,dtype = rc.dtype)
    result_img = cv2.merge((rc,gc,bc,ac))
    result_img = result_img * pred_mask[...,np.newaxis]

    display([img, pred, result_img],['Input Image', 'Predict Mask', 'Remove BG'])
    