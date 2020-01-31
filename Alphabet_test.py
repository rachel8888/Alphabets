import tensorflow as tf
import numpy as np
import cv2

def Loading_Model():
    new_model = tf.keras.models.load_model('Alphabet.model')
    return new_model

def Resizing_Image():
    img_array = cv2.imread("K.png", cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(img_array, (28, 28))
    resized_image = tf.cast(resized_image, tf.float32)
    return resized_image

def Reshaping():
    resized_image = Resizing_Image()
    t = []
    t.append(resized_image)
    t = np.asarray(t)
    u = []
    u.append(t)
    u = np.asarray(u)
    for i in range (0,3):
        u = np.moveaxis(u, -1, 0)
    return u    

u = Reshaping()
new_model = tf.keras.models.load_model('Alphabet.model')
perdictions = new_model.predict(u)
print("Predicted Alphabet Label is:", end = '')
print(np.argmax(perdictions[0]))
