from .constants import DOWNLOADED_IMAGES_PATHS
from typing import List
import os 
import tensorflow as tf

def ensure_directories():
    os.makedirs(DOWNLOADED_IMAGES_PATHS, exist_ok=True)

def delete_files(filespath:List[str]):
    for file in filespath:
        if os.path.exists(file):
            os.remove(file)

def model_reconstruct():
    base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights=None,
    input_shape=(150,150,3)
)
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(150,150,3),name='input_layer')
    x = base_model(inputs,training = False)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling_layer')(x)
    x = tf.keras.layers.Dense(128,activation ='relu')(x)
    x=tf.keras.layers.Dropout(0.5)(x)
    outputs=tf.keras.layers.Dense(1,activation='sigmoid',name='output_layer')(x)
    model = tf.keras.Model(inputs,outputs)
    return model
