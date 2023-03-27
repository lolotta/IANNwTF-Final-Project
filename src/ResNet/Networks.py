import tensorflow as tf
from Features import Low_Level_Features, Mid_Level_Features, High_Level_Features
from Layers import Classification_Layers, Fusion_Layer, Colorization_Layers

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
    def __init__(self, low_level):
        super().__init__()
        self.low_level = low_level 
        self.high_level = High_Level_Features()
        self.classification = Classification_Layers()

    def call(self, input, training=False):
        low = self.low_level(input, training)
        high = self.high_level(low, training)
        label = self.classification(high, training)
        return high, label

 
