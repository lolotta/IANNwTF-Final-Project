import tensorflow as tf
from keras.layers import Dense, Conv2D, Reshape, GlobalAveragePooling2D, MaxPooling2D, UpSampling2D, Flatten


class Low_Level_Features(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(64, 3, activation='relu', padding='same', strides=1) 
        self.batchnorm_1 = tf.keras.layers.BatchNormalization()
        self.conv2 = Conv2D(128, 3, activation='relu', padding='same', strides=1)
        self.batchnorm_2 = tf.keras.layers.BatchNormalization()
        self.conv3 = Conv2D(128, 3, activation='relu', padding='same', strides=2)
        self.batchnorm_3 = tf.keras.layers.BatchNormalization() 
        self.conv4 = Conv2D(256, 3, activation='relu', padding='same', strides=1)
        self.batchnorm_4 = tf.keras.layers.BatchNormalization() 
        self.conv5 = Conv2D(256, 3, activation='relu', padding='same', strides=2)
        self.batchnorm_5 = tf.keras.layers.BatchNormalization() 
        self.conv6 = Conv2D(512, 3, activation='relu', padding='same', strides=1)
        self.batchnorm_6 = tf.keras.layers.BatchNormalization() 

    def __call__(self, x, training=False):
        x = self.conv1(x)
        x = self.batchnorm_1(x, training)
        x = self.conv2(x)
        x = self.batchnorm_2(x, training)
        x = self.conv3(x)
        x = self.batchnorm_3(x, training)
        x = self.conv4(x)
        x = self.batchnorm_4(x, training)
        x = self.conv5(x)
        x = self.batchnorm_5(x, training)
        x = self.conv6(x)
        x = self.batchnorm_6(x, training)

        return x
    

class Mid_Level_Features(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(512, 3, activation='relu', padding='same', strides=1) 
        self.batchnorm_1 = tf.keras.layers.BatchNormalization()

        self.conv2 = Conv2D(256, 3, activation='relu', padding='same', strides=1) 
        self.batchnorm_2 = tf.keras.layers.BatchNormalization()


    def __call__(self, x, training=False):
        x = self.conv1(x)
        x = self.batchnorm_1(x, training)

        x = self.conv2(x)
        x = self.batchnorm_2(x, training)

        return x
    

class High_Level_Features(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(512, 3, activation='relu', padding='same', strides=1)
        self.max_pool1 = MaxPooling2D()
        self.batchnorm_1 = tf.keras.layers.BatchNormalization() 
        self.conv2 = Conv2D(512, 3, activation='relu', padding='same', strides=1)
        self.batchnorm_2 = tf.keras.layers.BatchNormalization() 
        self.conv3 = Conv2D(512, 3, activation='relu', padding='same', strides=1)
        self.max_pool2 = MaxPooling2D()

        self.batchnorm_3 = tf.keras.layers.BatchNormalization() 
        self.conv4 = Conv2D(512, 3, activation='relu', padding='same', strides=1)
        self.batchnorm_4 = tf.keras.layers.BatchNormalization() 
        #self.flatten = Flatten()
        self.global_avg = GlobalAveragePooling2D()

        self.dense1 = Dense(1024, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = Dense(512, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.dense3 = Dense(256, activation="relu")
        self.dropout3 = tf.keras.layers.Dropout(0.2)

    def __call__(self, x, training=False):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.batchnorm_1(x, training)
        x = self.conv2(x)
        x = self.batchnorm_2(x, training)
        x = self.conv3(x)
        x = self.max_pool2(x)
        x = self.batchnorm_3(x, training)
        x = self.conv4(x)
        x = self.batchnorm_4(x, training)
        #x = self.flatten(x)
        x = self.global_avg(x)
        x = self.dense1(x)
        x = self.dropout1(x, training)
        x = self.dense2(x)
        x = self.dropout2(x, training)
        x = self.dense3(x)
        x = self.dropout3(x, training)


        return x
    
class Classification_Layers(tf.keras.Model):
    def __init__(self, no_of_classes):
        super().__init__()
        self.dense1 = Dense(256, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = Dense(no_of_classes, activation="softmax")

    def __call__(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout1(x, training)

        x = self.dense2(x)
        return x
    
class Fusion_Layer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        #32,256,256
        #

        self.repeat_layer = tf.keras.layers.RepeatVector(32*32)
        self.reshape = tf.keras.layers.Reshape(([32,32,256]))
        self.concat = tf.keras.layers.Concatenate(axis=3)
        self.conv = Conv2D(256, kernel_size=1,strides=1, activation="relu", padding="same")


    def __call__(self, mid_level, global_vector, training=False):
        x = self.repeat_layer(global_vector) 
        x = self.reshape(x)
        x = self.concat([mid_level, x]) 
        x = self.conv(x)

        return x 
    

class Colorization_Layers(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(128, 3, activation='relu', padding='same', strides=1) 
        self.upsampling1 = UpSampling2D(2)
        self.batchnorm_1 = tf.keras.layers.BatchNormalization() 
        self.conv2 = Conv2D(64, 3, activation='relu', padding='same', strides=1) 
        self.conv3 = Conv2D(64, 3, activation='relu', padding='same', strides=1)
        self.upsampling2 = UpSampling2D(2) 
        self.batchnorm_2 = tf.keras.layers.BatchNormalization() 
        self.conv4 = Conv2D(32, 3, activation='relu', padding='same', strides=1)
        self.conv5 = Conv2D(2, 3, activation='sigmoid', padding='same', strides=1) 
        self.upsampling3 = UpSampling2D(2) 


    def __call__(self, input, training=False):
        x = self.conv1(input)
        x = self.upsampling1(x)
        x = self.batchnorm_1(x, training)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.upsampling2(x)
        x = self.batchnorm_2(x, training)
        x = self.conv4(x)
        x = self.conv5(x)
        #x = self.upsampling3(x)

        return x

class Colorization_Network(tf.keras.Model):
    def __init__(self, low_level):
        super().__init__()
        self.low_level = low_level
        self.mid_level = Mid_Level_Features()
        self.fusion = Fusion_Layer()
        self.colorization = Colorization_Layers()


    def call(self, input, high_level_input, training=False):
        low = self.low_level(input, training)
        middle = self.mid_level(low, training)
        fused = self.fusion(middle, high_level_input, training)
        colored = self.colorization(fused, training)
        return colored


class Classification_Network(tf.keras.Model):
    def __init__(self, low_level, no_of_classes):
        super().__init__()
        self.low_level = low_level 
        self.high_level = High_Level_Features()

        self.classification = Classification_Layers(no_of_classes)


    def call(self, input, training=False):
        low = self.low_level(input, training)
        high = self.high_level(low, training)
        label = self.classification(high, training)
        return high, label

 
