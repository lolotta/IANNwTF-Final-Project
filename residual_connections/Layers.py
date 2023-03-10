import tensorflow as tf
from keras.layers import Dense, Conv2D, Reshape, GlobalAveragePooling2D, MaxPooling2D, UpSampling2D, Flatten

class Classification_Layers(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(256, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = Dense(20, activation="softmax")

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
    
    
class ResidualConnectedCNNLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(ResidualConnectedCNNLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', activation='relu')

    def call(self, x):
        c = self.conv(x)
        x = c+x
        return x

class ResidualConnectedCNNBlock(tf.keras.layers.Layer):
    def __init__(self, depth, layers):
        super(ResidualConnectedCNNBlock, self).__init__()
        self.deeper_layer = tf.keras.layers.Conv2D(filters=depth, kernel_size=3, padding='same', activation='relu')
        self.layers = [ResidualConnectedCNNLayer(depth) for _ in range(layers)]

    def call(self, x):
        x = self.deeper_layer(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
    
    
class Colorization_Layers(tf.keras.Model):
    def __init__(self):
        super().__init__()
        #self.residualblock1 = ResidualConnectedCNNBlock(128,3)
        self.conv1 = Conv2D(128, 3, activation='relu', padding='same', strides=1) 
        self.upsampling1 = UpSampling2D(2)
        self.batchnorm_1 = tf.keras.layers.BatchNormalization()
        #self.residualblock2 = ResidualConnectedCNNBlock(64,3)
        #self.residualblock3 = ResidualConnectedCNNBlock(64,3)
        self.conv2 = Conv2D(64, 3, activation='relu', padding='same', strides=1) 
        self.conv3 = Conv2D(64, 3, activation='relu', padding='same', strides=1)
        self.upsampling2 = UpSampling2D(2) 
        self.batchnorm_2 = tf.keras.layers.BatchNormalization() 
        #self.residualblock4 = ResidualConnectedCNNBlock(32,3)
        self.conv4 = Conv2D(32, 3, activation='relu', padding='same', strides=1)
        self.conv5 = Conv2D(2, 3, activation='sigmoid', padding='same', strides=1) 
        self.upsampling3 = UpSampling2D(2) 


    def __call__(self, input, training=False):
        #x = self.residualblock1(input)
        x = self.conv1(input)
        x = self.upsampling1(x)
        x = self.batchnorm_1(x, training)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.residualblock2(x)
        #x = self.residualblock3(x)
        x = self.upsampling2(x)
        x = self.batchnorm_2(x, training)
        #x = self.residualblock4(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #x = self.upsampling3(x)

        return x
