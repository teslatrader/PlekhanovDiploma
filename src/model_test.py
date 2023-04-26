import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from PIL import Image
import face_recognition as fr
import os
import pickle


# load the model from disk
model_name = 'model_logreg'
model = pickle.load(open(model_name, 'rb'))

df_person_ids_name = 'df_person_id.pkl'
df_person_ids = pd.read_pickle(df_person_ids_name)
print(df_person_ids)


def image_prepocessing(image: str, side_size=256):
    try:
        img = Image.open(image)
        img.show()
        img = img.convert('RGB')

        # normlization and resize to single side size (by bigger side)
        size = img.size
        value_max = max(size)
        normalization_ratio = round(value_max / side_size, 2)
        width_normalized = int(size[0] / normalization_ratio)
        height_normalized = int(size[1] / normalization_ratio)
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, [height_normalized, width_normalized])

        # save original image
        img_name = image.replace('.', '_resized.')
        tf.keras.utils.save_img(img_name, img)
        return img_name

    except Exception as e:
        print(f'During image normalization we got exception: \n{e}')
        os.remove(image)


def get_embedding(image: str):
    img = fr.load_image_file(image)
    face_location = fr.face_locations(img, number_of_times_to_upsample=1)
    qty_of_faces = len(face_location)
    if qty_of_faces == 1:
        # creating face embedding
        face_embedding = fr.face_encodings(img)[0]
        face_embedding = face_embedding.tolist()

        # create dataframe with embedding to predict
        columns_name = [f'x_{idx}' for idx in range(128)]
        df_test = pd.DataFrame(columns=columns_name)
        df_test.loc[len(df_test)] = face_embedding
        return df_test
    elif qty_of_faces < 1:
        print('Face was not found, please reload the photo with only the single face.')
    else:
        print('Was found more than one face, please, reload the photo with only single face.')


def match_person(df_matches, person_id):
    for per_id in df_matches['person_id']:
        if per_id == person_id:
            return df_matches.loc[per_id]['person']


def get_random_photo(person_name):
    path = f'../inputs/{person_name}'
    imgs_list = os.listdir(path)
    qty_of_imgs = len(imgs_list)
    img_idx = np.random.randint(0, qty_of_imgs-1, size=1)[0]
    img_name = f'{path}/{imgs_list[img_idx]}'
    img = Image.open(img_name)
    img.show()


def get_prediction(image_name: str):
    img_test = image_prepocessing(image_name)
    embedding = get_embedding(img_test)
    print(f'Embedding of test photo:\n{embedding}')

    prediction = model.predict(embedding)
    print(f'Person id predicted = {prediction}')

    y_pred_probability = model.predict_proba(embedding)
    print(f'Probability scores by classes:\n{y_pred_probability}')

    person_name = match_person(df_person_ids, prediction[0])
    print(f'Yours highest SUPERSTAR likelihood with {person_name}')
    get_random_photo(person_name)


if __name__ == '__main__':
    my_face_test = 'FK_test.jpg'
    chris_hem_test = '_Chris_Hem_test.jpg'
    get_prediction(my_face_test)
    get_prediction(chris_hem_test)