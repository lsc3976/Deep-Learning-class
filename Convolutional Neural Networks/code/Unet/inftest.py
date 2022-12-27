import numpy as np
import tensorflow as tf
import data as dt
from display import show_predictions

def inference_test():
    """
    train된 모델의 test 진행
    """
    path = "./dataset"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = dt.load_data(path)

    test_dataset = dt.tf_dataset(test_x, test_y, batch=32)
    model = tf.keras.models.load_model('model.h5')
    show_predictions(model,test_dataset,6)

