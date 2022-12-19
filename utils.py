import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight

import seaborn as sns

def store_idx_classid(ds, path):
    class_names = list(ds.class_names)
    dic={}
    idxs, vals = [], []
    for idx, val in enumerate(class_names):
        idxs.append(idx)
        vals.append(val)
        
    dic['index'] = idxs
    dic['name'] = vals
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(path+'idx_class.csv', index=False)

def get_class_weight(ds):
    y = np.concatenate([y for x, y in ds], axis=0)
    class_weights = class_weight.compute_class_weight('balanced',
                                                 classes = np.unique(y),
                                                 y = y)
    return class_weights
    

def get_ds(data_dir, save_path, image_size = 8, batch_size = 32):
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.1, subset="training", 
                                                       seed=123, image_size=(image_size, image_size), 
                                                       batch_size=batch_size)
    store_idx_classid(train_ds, save_path)
    num_classes = len(train_ds.class_names)
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.1, subset="validation", 
                                                        seed=123, image_size=(image_size, image_size),
                                                        batch_size=batch_size)

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, num_classes

def plot(history):
    plt.figure(1)
    plt.title('Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()

    plt.figure(2)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.show()

def get_test_acc(dir, model,image_size=8):
    test_path = f"{dir}/test/"
    test_df = pd.read_csv(f"{test_path}Truth.csv")
    test_df.head()

    arr = []
    for image in test_df['image'].values:
        img_path = f"{test_path}{image}"

        image = tf.keras.preprocessing.image.load_img(img_path, 
                                                        target_size=(image_size,image_size))
        input_arr = tf.keras.utils.img_to_array(image)
        input_arr = input_arr*(1./255)
        arr.append(input_arr)


    input_arr= np.array([arr]).reshape(-1,image_size,image_size,3)
    y_pred = model.predict(input_arr)
    y_pred = np.argmax(y_pred, axis = 1)
    y_actual = list(test_df['superclass_index'])

    print(classification_report(y_actual, y_pred))
    print("Accuracy :", accuracy_score(y_actual, y_pred))

    cf_matrix = confusion_matrix(y_actual, y_pred)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap='Blues')

def get_test_subsmission(model, model_name, image_size):
    test_path = f"Realesed Data/test_shuffle/"

    arr = []
    image_names = []
    for image in os.listdir(test_path):
        img_path = f"{test_path}{image}"
        image_names.append(image)
        image = tf.keras.preprocessing.image.load_img(img_path, 
                                                      target_size=(image_size,image_size))
        input_arr = tf.keras.utils.img_to_array(image)
        input_arr = input_arr * (1.0/255)
        arr.append(input_arr)

    input_arr= np.array(arr)
    input_arr= np.array([arr]).reshape(-1,image_size,image_size,3)
    y_pred = model.predict(input_arr)
    y_pred = np.argmax(y_pred, axis = 1)
    dic = {'img':image_names , 'prediction_index':y_pred}
    df = pd.DataFrame.from_dict(dic)
    df['img'] = df['img'].apply(lambda x: int(x.replace('.jpg','')))
    df = df.sort_values('img').reset_index(drop=True)

    map = pd.read_csv('Realesed Data/super_classes_mapping.csv')
    your_pred_list = df['prediction_index'].tolist()

    map_dict = {}
    for index , row in map.iterrows():
        map_dict[index] = row[1]

    submission = []
    for index in your_pred_list:
        submission.append(map_dict[index])

    res = pd.DataFrame({'predictions': submission})
    res.head()
    res.to_csv(f'{model_name}.csv', index=False)