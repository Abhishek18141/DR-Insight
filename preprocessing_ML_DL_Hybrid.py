import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
import tensorflow as tf

# DL Preprocessing (Binary & Multiclass)
def preprocess_dl_image(image):
    # Load InceptionV3 model (shared for feature extraction)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    feature_extractor_model = Model(inputs=base_model.input, outputs=x)

    # Preprocess the image
    img = image.resize((299, 299))
    img_array = np.expand_dims(img_to_array(img), axis=0)
    img_array = preprocess_input(img_array)

    # Extract features
    return feature_extractor_model.predict(img_array).flatten().reshape(1, -1)

# Hybrid Binary Preprocessing (MobileNetV2 + MLP)
def preprocess_hybrid_binary_image(image):
    # Load MobileNetV2 as feature extractor
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    feature_extractor_model = Model(inputs=base_model.input, outputs=x)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)  

    # Extract features
    return feature_extractor_model.predict(img_array).flatten().reshape(1, -1)

# Hybrid Multiclass Preprocessing (InceptionV3 + MLP)
def preprocess_hybrid_multiclass_image(image):
    # Load InceptionV3 as feature extractor
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    feature_extractor_model = Model(inputs=base_model.input, outputs=x)

    # Preprocess the image
    img = image.resize((299, 299))
    img_array = np.expand_dims(img_to_array(img), axis=0)
    img_array = preprocess_input(img_array)  

    # Extract features
    return feature_extractor_model.predict(img_array).flatten().reshape(1, -1)
