import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2

H = 256
W = 256


def process_data(data_path, file_path):
    """
    data path를 list로 return
    """
    df = pd.read_csv(file_path, sep=" ", header=None, skiprows=range(6))
    names = df[0].values

    images = [os.path.join(data_path, f"images/{name}.jpg") for name in names]
    masks = [os.path.join(
        data_path, f"annotations/trimaps/{name}.png") for name in names]

    return images, masks


def load_data(path):
    """
    process_data를 통해 얻은 path list를 train, test, valid set으로 분리
    """
    data_path = os.path.join(path,"annotations/list.txt")

    train_x, train_y = process_data(path,data_path)

    train_x, valid_x = train_test_split(train_x, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=0.2, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=0.25, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=0.25, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image(image_path):
    """ 
    image를 opencv로 읽어와 resize와 normalization 진행
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (H, W))
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def read_mask(mask_path):
    """
    mask(true mask)를 opencv로 읽어와 resize 진행 후, 값을 1 감소
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (H, W))
    mask = mask - 1
    mask = mask.astype(np.int32)
    return mask


def preprocess(x, y):
    """
    image, mask path를 read 해서 return
    """
    def read_func(image_path, mask_path):
        """
        이미지 주소를 문자열로 변환 후 해당 이미지를 읽어옴
        """
        image_path = image_path.decode()
        mask_path = mask_path.decode()

        image = read_image(image_path)
        mask = read_mask(mask_path)

        return image, mask

    image, mask = tf.numpy_function(read_func, [x, y], [tf.float32, tf.int32])
    image.set_shape([H, W, 3])
    mask.set_shape([H,W])
    return image, mask


def tf_dataset(x, y, batch=32):
    """
    create dataset
    """

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset
