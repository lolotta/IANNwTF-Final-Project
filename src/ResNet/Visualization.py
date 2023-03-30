import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

import matplotlib.pyplot as plt


def visualize(predicted_color, predicted_label, data, class_names):
    images = data[0]
    labels = data[1]
    
    
    grey_images = images[0]
    grey_image = grey_images[0]
    color_images = images[1]
    color_image = color_images[0]
    label = labels[0]

    
    # manually clipping values, because Lab color space is weird
    predicted_color_lab_scaled = tf.clip_by_value(((predicted_color - [0.5, 0.5]) * [200, 200]), -100, 100)
    color_image_lab_scaled = tf.clip_by_value(((color_image - [0.5, 0.5]) * [200, 200]), -100, 100)
    greyscale =tf.squeeze((grey_image - [0.5]) * [200], axis=-1)


    rgb_prediction = tfio.experimental.color.lab_to_rgb(tf.stack([greyscale, predicted_color_lab_scaled[:,:,0], predicted_color_lab_scaled[:,:,1]],axis=-1))
    rgb_original = tfio.experimental.color.lab_to_rgb(tf.stack([greyscale, color_image_lab_scaled[:,:,0], color_image_lab_scaled[:,:,1]], axis=-1))

    #labels
    predicted_label = class_names[tf.argmax(predicted_label).numpy()]
    true_label = class_names[tf.argmax(label).numpy()]

    print(predicted_label, true_label)


    fig, ax = plt.subplots(1, 3, figsize = (18, 30))

    greyscale = ((greyscale/200) + 0.5) * 255
    ax[0].imshow(greyscale, cmap="gist_gray") 
    ax[0].axis('off')
    ax[0].set_title('greyscale')

    ax[1].imshow(rgb_prediction) 
    ax[1].axis('off')
    ax[1].set_title('pred: ' + predicted_label)
    
    ax[2].imshow(rgb_original) 
    ax[2].axis('off')
    ax[2].set_title('orig: ' + true_label)


