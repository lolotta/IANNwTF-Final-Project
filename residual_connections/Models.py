import tensorflow as tf
from Features import Low_Level_Features, Mid_Level_Features, High_Level_Features
from Layers import Classification_Layers, Fusion_Layer, Colorization_Layers
from Networks import  Colorization_Network, Classification_Network


class Only_Colorization_Model(tf.keras.Model): 
    def __init__(self, optimizer, loss_function_color):
        super().__init__()
        self.low_level = Low_Level_Features()
        self.mid_level = Mid_Level_Features()
        self.colorization = Colorization_Layers()


        self.metrics_list = [
            tf.keras.metrics.Mean(name="loss_color"),
        ]

        self.optimizer = optimizer
        self.loss_function_color = loss_function_color

        

    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()


    def call(self, input, training=False):
        low = self.low_level(input, training)
        middle = self.mid_level(low, training=training)
        colored = self.colorization(middle, training)
        return colored


    @tf.function
    def train_step(self, data):
        images,  label = data
        grey_image, color_image = images
        with tf.GradientTape() as color_tape: 
            predicted_color = self(grey_image, training = True)
            loss_color = self.loss_function_color(color_image, predicted_color)

        gradients_color = color_tape.gradient(loss_color, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients_color, self.trainable_variables))
        self.metrics[0].update_state(loss_color)  
        return predicted_color

    @tf.function
    def test_step(self, data):
        images, label = data
        grey_image, color_image = images    
        predicted_color = self(grey_image, training = False)
        loss_color = self.loss_function_color(color_image, predicted_color)
        self.metrics[0].update_state(loss_color)  
        return predicted_color
    
    
    
class Only_Classification_Model(tf.keras.Model):
    def __init__(self, optimizer, loss_function_category):
        super().__init__()
        self.low_level = Low_Level_Features()        
        self.classification_model = Classification_Network(self.low_level)

        self.metrics_list = [
            tf.keras.metrics.Mean(name="loss_category"),
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5,name="top-5-accuracy")]

        self.optimizer = optimizer
        self.loss_function_category = loss_function_category

    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()

    def call(self, input, training=False):
        _, label = self.classification_model(input, training)
        return label

    @tf.function
    def train_step(self, data):
        images,  label = data
        grey_image, color_image = images
        with tf.GradientTape() as class_tape: 
            predicted_label = self(grey_image, training = True)
            loss_category = self.loss_function_category(label, predicted_label) / 200   

        gradients_category = class_tape.gradient(loss_category, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients_category, self.trainable_variables))
        self.metrics[0].update_state(loss_category)  
        self.metrics[1].update_state(label, predicted_label)
        self.metrics[2].update_state(label, predicted_label)

        return predicted_label

    @tf.function
    def test_step(self, data):
        images, label = data
        grey_image, color_image = images    
        predicted_label = self(grey_image, training = False)
        loss_category  = self.loss_function_category(label, predicted_label) / 200              
        self.metrics[0].update_state(loss_category)  
        self.metrics[1].update_state(label, predicted_label)
        self.metrics[2].update_state(label, predicted_label)

        return predicted_label
    
    
class Model(tf.keras.Model):
    def __init__(self, optimizer, loss_function_color, loss_function_category):
        super().__init__()
        self.low_level = Low_Level_Features()        

        self.colorization_model = Colorization_Network(self.low_level)
        self.classification_model = Classification_Network(self.low_level)

        self.metrics_list = [
            tf.keras.metrics.Mean(name="loss_color"),
            tf.keras.metrics.Mean(name="loss_category"),
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5,name="top-5-accuracy")]

        self.optimizer = optimizer
        self.loss_function_color = loss_function_color
        self.loss_function_category = loss_function_category

    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()

    def call(self, input, training=False):
        high_level_info, label = self.classification_model(input, training)
        colored = self.colorization_model(input, high_level_info, training)
        return colored, label

    @tf.function
    def train_step(self, data):
        images,  label = data
        grey_image, color_image = images
        with tf.GradientTape() as color_tape, tf.GradientTape() as class_tape: 
            predicted_color, predicted_label = self(grey_image, training = True)
            loss_color = self.loss_function_color(color_image, predicted_color)
            loss_category = self.loss_function_category(label, predicted_label) / 200

        gradients_color = color_tape.gradient(loss_color, self.colorization_model.trainable_variables)
        gradients_category = class_tape.gradient(loss_category, self.classification_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients_color, self.colorization_model.trainable_variables))
        self.optimizer.apply_gradients(zip(gradients_category, self.classification_model.trainable_variables))
        self.metrics[0].update_state(loss_color)  
        self.metrics[1].update_state(loss_category)
        self.metrics[2].update_state(label, predicted_label)
        self.metrics[3].update_state(label, predicted_label)
        
        return predicted_color, predicted_label

    @tf.function
    def test_step(self, data):
        images, label = data
        grey_image, color_image = images    
        predicted_color, predicted_label = self(grey_image, training = False)
        loss_color = self.loss_function_color(color_image, predicted_color)
        loss_category  = self.loss_function_category(label, predicted_label) / 200          
        self.metrics[0].update_state(loss_color)  
        self.metrics[1].update_state(loss_category)  
        self.metrics[2].update_state(label, predicted_label)
        self.metrics[3].update_state(label, predicted_label)

        return predicted_color, predicted_label
    
