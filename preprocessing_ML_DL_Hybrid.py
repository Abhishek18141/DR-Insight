import numpy as np
from skimage import color, transform, exposure, filters
from skimage.feature import hog
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
import tensorflow as tf

# ML Preprocessing (Binary & Multiclass)
def preprocess_ml_image(image):
    # Convert to grayscale
    gray_image = color.rgb2gray(image)
    # Resize the image
    resized_image = transform.resize(gray_image, (256, 256), anti_aliasing=True)
    # Enhance contrast
    enhanced_image = exposure.equalize_adapthist(resized_image, clip_limit=0.03)
    # Remove noise
    denoised_image = filters.gaussian(enhanced_image, sigma=1)
    # Normalize the image
    preprocessed_image = denoised_image / denoised_image.max()
    # Extract HOG features
    features, _ = hog(
        preprocessed_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(3, 3),
        block_norm='L2-Hys',
        visualize=True
    )
    # Flatten and reshape for prediction
    return features.flatten().reshape(1, -1)

# DL Preprocessing (Binary & Multiclass)
def preprocess_dl_image(image):
    # Load InceptionV3 model (shared for feature extraction)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  # Freeze the base model

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    feature_extractor_model = tf.keras.Model(inputs=base_model.input, outputs=x)

    # Load and preprocess the image
    img = image.resize((299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

    # Extract features
    feature_vector = feature_extractor_model.predict(img_array)
    return feature_vector.flatten().reshape(1, -1)

# Hybrid Binary Preprocessing (MobileNetV2 + MLP)
def preprocess_hybrid_binary_image(image):
    # Step 1: Load the MobileNetV2 model as a feature extractor
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    feature_extractor_model = Model(inputs=base_model.input, outputs=x)

    # Step 2: Preprocess the input image
    img = image.resize((224, 224))  # Resize image to MobileNetV2 input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

    # Step 3: Extract features
    features = feature_extractor_model.predict(img_array)
    features = features.flatten().reshape(1, -1)  # Reshape for model input

    return features

# Hybrid Multiclass Preprocessing (InceptionV3 + MLP)
def preprocess_hybrid_multiclass_image(image):
    # Step 1: Load the feature extractor model (InceptionV3 with GlobalAveragePooling2D)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  # Freeze the base model
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    feature_extractor_model = Model(inputs=base_model.input, outputs=x)

    # Step 2: Preprocess the input image
    img = image.resize((299, 299))  # Resize image to InceptionV3 input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)  # Preprocess for InceptionV3

    # Step 3: Extract features
    features = feature_extractor_model.predict(img_array)
    features = features.flatten().reshape(1, -1)  # Reshape features for model input

    return features
