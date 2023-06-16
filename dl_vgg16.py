import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import threading
import os

def similarity(path1, path2):
    # Load the two images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    # Resize the images to 224x224
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))

    # Preprocess the images for the VGG16 model
    img1 = preprocess_input(img1)
    img2 = preprocess_input(img2)

    # Load the pre-trained VGG16 model
    base_model = VGG16(weights='imagenet')

    # Create a new model that outputs the last convolutional layer
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)

    # Extract the features from the images using the VGG16 model
    features1 = model.predict(np.expand_dims(img1, axis=0))
    features2 = model.predict(np.expand_dims(img2, axis=0))

    # Compute the Euclidean distance between the feature vectors
    distance = np.linalg.norm(features1 - features2)
    print(f'Distance: {distance} {path1} ===> {path2}')
    # return distance

def similarity_thread(imgs):
    threads = []

    for img in imgs:
        t = threading.Thread(target=similarity, args=(img[0], img[1]))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()