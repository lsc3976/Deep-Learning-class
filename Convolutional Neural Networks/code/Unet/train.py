import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import data as dt
import model
from tensorflow.keras.callbacks import ModelCheckpoint
from display import display


def train():
    """
    model train
    """
    np.random.seed(42)
    tf.random.set_seed(42)

    path = 'dataset/'
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = dt.load_data(path)

    shape = (256, 256, 3)
    num_classes = 3
    n_filters = 64
    epochs = 40
    buffer_size = 1000
    batch_size = 64
    unet = model.unet_model()

    unet.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['sparse_categorical_accuracy'])

    train_dataset = dt.tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = dt.tf_dataset(valid_x, valid_y, batch=batch_size)

    train_steps = len(train_x)//batch_size
    valid_steps = len(valid_x)//batch_size

    callbacks = [
        ModelCheckpoint("model.h5", verbose=1, save_best_only=True, monitor='val_loss')
    ]

    model_history = unet.fit(train_dataset, steps_per_epoch=train_steps, validation_data=valid_dataset,
                             validation_steps=valid_steps, epochs=epochs, callbacks=callbacks)

    plt.plot(model_history.history['loss'],label='loss')
    plt.plot(model_history.history['val_loss'],label='val_loss')
    plt.legend(loc='best')
    plt.show()