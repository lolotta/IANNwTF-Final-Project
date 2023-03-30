# training loop

import tensorflow as tf
from tqdm import tqdm

def training_loop(model, train_ds, test_ds, epochs, train_summary_writer, test_summary_writer, save_path):
    for epoch in range(epochs):
        model.reset_metrics()

        for data in tqdm(train_ds, position=0, leave=True):
            predicted_color, predicted_label = model.train_step(data)

        with train_summary_writer.as_default():
            tf.summary.scalar(model.metrics[0].name, model.metrics[0].result(), step=epoch)
            tf.summary.scalar(model.metrics[1].name, model.metrics[1].result(), step=epoch)
            tf.summary.scalar(model.metrics[2].name, model.metrics[2].result(), step=epoch)
            tf.summary.scalar(model.metrics[3].name, model.metrics[3].result(), step=epoch)

            
            
        print("Epoch: ", epoch+1)
        print("Loss Color: ", model.metrics[0].result().numpy(), "(Train)")
        print("Loss Category: ", model.metrics[1].result().numpy(), "(Train)")
        print("Accuracy: ", model.metrics[2].result().numpy(), "(Train)")
        print("Top-5-Accuracy: ", model.metrics[3].result().numpy(), "(Train)")


        model.reset_metrics()

        for data in tqdm(test_ds, position=0, leave=True):
            predicted_color, predicted_label = model.test_step(data)

        with test_summary_writer.as_default():
            tf.summary.scalar(model.metrics[0].name, model.metrics[0].result(), step=epoch)
            tf.summary.scalar(model.metrics[1].name, model.metrics[1].result(), step=epoch)
            tf.summary.scalar(model.metrics[2].name, model.metrics[2].result(), step=epoch)
            tf.summary.scalar(model.metrics[3].name, model.metrics[3].result(), step=epoch)
                    
        print("Loss Color: ", model.metrics[0].result().numpy(), "(Test)")
        print("Loss Category: ", model.metrics[1].result().numpy(), "(Test)")
        print("Accuracy: ", model.metrics[2].result().numpy(), "(Test)")
        print("Top-5-Accuracy: ", model.metrics[3].result().numpy(), "(Test)")




def training_loop_colorization(model, train_ds, test_ds, epochs, train_summary_writer, test_summary_writer, save_path):
    for epoch in range(epochs):
        model.reset_metrics()

        
        for data in tqdm(train_ds, position=0, leave=True):
            predicted_color = model.train_step(data)


        with train_summary_writer.as_default():
            tf.summary.scalar(model.metrics[0].name, model.metrics[0].result(), step=epoch)
        
        print("Epoch: ", epoch+1)
        print("Loss Color: ", model.metrics[0].result().numpy(), "(Train)")
        model.reset_metrics()

        last_data = None
        for data in tqdm(test_ds, position=0, leave=True):
            predicted_color = model.test_step(data)
            last_data = data


        with test_summary_writer.as_default():
            tf.summary.scalar(model.metrics[0].name, model.metrics[0].result(), step=epoch)
            
        print("Loss Color: ", model.metrics[0].result().numpy(), "(Test)")

def training_loop_classification(model, train_ds, test_ds, epochs, train_summary_writer, test_summary_writer, save_path):
    for epoch in range(epochs):
        model.reset_metrics()

        
        for data in tqdm(train_ds, position=0, leave=True):
            predicted_label = model.train_step(data)


        with train_summary_writer.as_default():
            tf.summary.scalar(model.metrics[0].name, model.metrics[0].result(), step=epoch)
            tf.summary.scalar(model.metrics[1].name, model.metrics[1].result(), step=epoch)
            tf.summary.scalar(model.metrics[2].name, model.metrics[2].result(), step=epoch)
        
        print("Epoch: ", epoch+1)
        print("Loss Category: ", model.metrics[0].result().numpy(), "(Train)")
        print("Accuracy: ", model.metrics[1].result().numpy(), "(Train)")
        print("Top-5-Accuracy: ", model.metrics[2].result().numpy(), "(Train)")

        model.reset_metrics()

        last_data = None
        for data in tqdm(test_ds, position=0, leave=True):
            predicted_label = model.test_step(data)
            last_data = data

        with test_summary_writer.as_default():
            tf.summary.scalar(model.metrics[0].name, model.metrics[0].result(), step=epoch)
            tf.summary.scalar(model.metrics[1].name, model.metrics[1].result(), step=epoch)
            tf.summary.scalar(model.metrics[2].name, model.metrics[2].result(), step=epoch)
            
        print("Loss Category: ", model.metrics[0].result().numpy(), "(Test)")
        print("Accuracy: ", model.metrics[1].result().numpy(), "(Test)")
        print("Top-5-Accuracy: ", model.metrics[2].result().numpy(), "(Test)")



