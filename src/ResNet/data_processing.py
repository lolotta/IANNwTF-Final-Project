import tensorflow as tf
import tensorflow_io as tfio
from keras.layers import Dense, Conv2D, Reshape, GlobalAveragePooling2D, MaxPooling2D, UpSampling2D, Flatten


def processing(directory):
    
    train_ds, test_ds = tf.keras.utils.image_dataset_from_directory(directory=directory, label_mode="categorical", 
    batch_size=None, shuffle=False, image_size=(128,128), validation_split=0.1, subset="both", crop_to_aspect_ratio=True)

    class_names = train_ds.class_names 
    print('classes: ', train_ds.class_names)
    
    # create training and testing datasets 
    train_data = preprocess_train(train_ds)
    print('data example: ' ,train_data)
    test_data = preprocess_test(test_ds)
    
    return train_data, test_data, class_names
    

def preprocess_train(data):
    
    # init layers 
    normalizing_layer = tf.keras.layers.Rescaling(1./255)
    flip_layer = tf.keras.layers.RandomFlip()
    rotation_layer = tf.keras.layers.RandomRotation((-0.1,0.1))   
    
    # process data
    data = data.map(lambda x,y: (normalizing_layer(x), y))
    data = data.map(lambda x,y: (flip_layer(x), y))
    data = data.map(lambda x,y: (rotation_layer(x), y))
    
    data = data.map(lambda x,y: (tfio.experimental.color.rgb_to_lab(x), tf.cast(y, tf.int32)))
    data = data.map(lambda image, label: ((((tf.expand_dims((image[:,:,0]/200 + 0.5), -1)), ((image[:,:,1:]/200) + 0.5))), label))

    data = data.shuffle(1000).batch(16).prefetch(tf.data.AUTOTUNE)

    return data


def preprocess_test(data):
    
    normalizing_layer = tf.keras.layers.Rescaling(1./255)
     
    data = data.map(lambda x,y: (normalizing_layer(x), y))
    data = data.map(lambda x,y: (tfio.experimental.color.rgb_to_lab(x), tf.cast(y, tf.int32)))

    data = data.map(lambda image, label: (((tf.expand_dims((image[:,:,0]/200 + 0.5), -1), ((image[:,:,1:]/200) + 0.5))), label))

    data = data.shuffle(1000).batch(16).prefetch(tf.data.AUTOTUNE)

    return data
