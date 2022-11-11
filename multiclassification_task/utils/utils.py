import splitfolders
import os
import tensorflow as tf
import pandas as pd
import numpy as np

METADATA = os.path.dirname(os.path.abspath(__file__)) + "/../metadata_validation.txt"

def split_folders(split_val: float, log=False, override_metadata=False, seed=100):
    main_path = os.path.dirname(os.path.abspath(__file__))
    if "metadata_validation.txt" in os.listdir(f"{main_path}/.."):
        with open(METADATA, "r") as reader:
            content = reader.readline()
            past_validation_split = content.split(" ")[-1]
            if past_validation_split == split_val and not override_metadata:
                if log:
                    print("Dataset already splitted")
                return
    data_path = f"{main_path}/../training_data_final/"
    splitfolders.ratio(data_path, output=f"{main_path}/../", ratio=(1 - split_val, split_val, 0), seed=seed)
    with open(METADATA, "w") as writer:
        writer.write(f"current validation split: {split_val}")

def prepare_batches(training_dir: str, validation_dir: str, image_shape: tuple, batch_size: tuple):
    """
    create two Dataset objects for training and validation
    """
    autotune = tf.data.AUTOTUNE
    train = tf.keras.utils.image_dataset_from_directory(training_dir,
                                                               labels="inferred",
                                                               label_mode="categorical",
                                                               batch_size=batch_size,
                                                               image_size=image_shape,
                                                               shuffle=True
                                                               ).prefetch(buffer_size=autotune)
    valid =  tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                               labels="inferred",
                                                               label_mode="categorical",
                                                               batch_size=batch_size,
                                                               image_size=image_shape,
                                                               shuffle=True
                                                               ).prefetch(buffer_size=autotune)

    return train, valid


def build_heatmap(model, validation_generator):
    y_predicted = model.predict(validation_generator)
    y_predicted = tf.argmax(y_predicted, axis=1)
    y_test_labels = tf.keras.utils.to_categorical(
        validation_generator.labels, num_classes=None, dtype='float32'
    )
    y_test_labels = tf.argmax(y_test_labels, axis=1)
    confusion_matrix = tf.math.confusion_matrix(
        y_test_labels, 
        y_predicted,
        num_classes=8
    )
    c = []
    for item in confusion_matrix:
        c.append(np.around(item / np.sum(item), decimals=3))
    df_heatmap = pd.DataFrame(c)
    return df_heatmap

if __name__ == "__main__":
    split_folders(0.8)
