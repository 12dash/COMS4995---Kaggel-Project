import os
import numpy as np
import tensorflow as tf

def get_test_subsmission(image_size):
    test_path = f"Realesed Data/test_shuffle/"

    arr = []
    image_names = []
    for image in os.listdir(test_path):
        img_path = f"{test_path}{image}"
        image_names.append(image)
        image = tf.keras.preprocessing.image.load_img(img_path, 
                                                      target_size=(image_size,image_size))
        input_arr = tf.keras.utils.img_to_array(image)
        input_arr = input_arr *(1.0/255)
        arr.append(input_arr)

    input_arr= np.array(arr)
    input_arr= np.array([arr]).reshape(-1,image_size,image_size,3)
    return input_arr, image_names