import tensorflow as tf
from Layers import ResidualConnectedCNNBlock
from keras.layers import Dense, Conv2D, Reshape, GlobalAveragePooling2D, MaxPooling2D, UpSampling2D, Flatten


class Low_Level_Features(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.residualblock1 = ResidualConnectedCNNBlock(64, 3) 
        self.batchnorm_1 = tf.keras.layers.BatchNormalization()
        self.residualblock2 = ResidualConnectedCNNBlock(128, 3)
        self.batchnorm_2 = tf.keras.layers.BatchNormalization()
        self.residualblock3 = ResidualConnectedCNNBlock(128, 3)
        self.batchnorm_3 = tf.keras.layers.BatchNormalization() 
        self.residualblock4 = ResidualConnectedCNNBlock(256, 3)
        self.batchnorm_4 = tf.keras.layers.BatchNormalization() 
        self.residualblock5 = ResidualConnectedCNNBlock(256, 3)
        self.batchnorm_5 = tf.keras.layers.BatchNormalization() 
        self.residualblock6 = ResidualConnectedCNNBlock(512, 3)
        # pooling layer - more of these? 
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.batchnorm_6 = tf.keras.layers.BatchNormalization() 

    def __call__(self, x, training=False):
        x = self.residualblock1(x)
        x = self.batchnorm_1(x, training)
        x = self.residualblock2(x)
        x = self.batchnorm_2(x, training)
        x = self.residualblock3(x)
        x = self.batchnorm_3(x, training)
        x = self.residualblock4(x)
        x = self.batchnorm_4(x, training)
        x = self.residualblock5(x)
        x = self.pooling(x)
        x = self.residualblock6(x)
        x = self.batchnorm_6(x, training)

        return x
    
    
    
class Mid_Level_Features(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.residualblock1 = ResidualConnectedCNNBlock(512, 3)
        self.batchnorm_1 = tf.keras.layers.BatchNormalization() 
        # pooling layer
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.residualblock2 = ResidualConnectedCNNBlock(256, 3)
        self.batchnorm_2 = tf.keras.layers.BatchNormalization() 
        

    def __call__(self, x, training=False):
        x = self.residualblock1(x)
        x = self.batchnorm_1(x)
        x = self.pooling(x)
        x = self.residualblock2(x)
        x = self.batchnorm_2(x)
       
        return x
    
    
    
class High_Level_Features(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.residualblock1 = ResidualConnectedCNNBlock(512, 3)
        self.max_pool1 = MaxPooling2D()
        self.batchnorm_1 = tf.keras.layers.BatchNormalization() 
        self.residualblock2 = ResidualConnectedCNNBlock(512, 3)
        self.batchnorm_2 = tf.keras.layers.BatchNormalization() 
        self.residualblock3 = ResidualConnectedCNNBlock(512, 3)
        self.max_pool2 = MaxPooling2D()
        
        self.batchnorm_3 = tf.keras.layers.BatchNormalization() 
        self.residualblock4 = ResidualConnectedCNNBlock(512, 3)
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
        x = self.residualblock1(x)
        x = self.max_pool1(x)
        x = self.batchnorm_1(x, training)
        x = self.residualblock2(x)
        x = self.batchnorm_2(x, training)
        x = self.residualblock3(x)
        x = self.max_pool2(x)
        x = self.batchnorm_3(x, training)
        x = self.residualblock4(x)
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