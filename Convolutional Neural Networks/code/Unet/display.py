from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

def display(display_list,title=['Input Image', 'True Mask', 'Predicted Mask']):
    plt.figure(figsize=(15, 15))
    title = title

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    """
    Create predicted mask
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]

    return pred_mask[0]


def show_predictions(model, dataset, num=1):
    """
    Displays the first image of each of the num batches
    """

    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        pred_mask = create_mask(pred_mask)
        pred_mask = np.reshape(pred_mask,(256,256))
        display([image[0], mask[0], pred_mask])