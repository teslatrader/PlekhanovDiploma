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


df_name = 'data_set.pkl'
df = pd.read_pickle(df_name)
print(df)

df_person_ids_name = 'df_person_id.pkl'
df_person_ids = pd.read_pickle(df_person_ids_name)
print(df_person_ids)

X = df.drop(['person name', 'target'], axis=1)
y = df['target']
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# print(X_train, '\n', y_train)
# corr = X_train.corr()
# plt.matshow(corr)
# columns_qty = range(len(corr.columns))
# plt.xticks(columns_qty, corr.columns)
# plt.yticks(columns_qty, corr.columns)
# plt.show()
# model = sm.Logit(y_train, sm.add_constant(X_train)).fit()
# print(model.summary())
model = LogisticRegression(multi_class='multinomial')
model.fit(X_train, y_train)
print(f'X_test\n{X_test}')
y_pred = model.predict(X_test)
f_1 = f1_score(y_test, y_pred, average='micro')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'f1 = {f_1}')
print(f'mae = {mae}')
print(f'mse = {mse}')
print(f'mape = {mape}')


def image_prepocessing(image: str, side_size=256):
    try:
        img = Image.open(image)
        # img.show()
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
        # creating face embedding and linking it with target person
        face_embedding = fr.face_encodings(img)[0]
        face_embedding = face_embedding.tolist()
        # face_embedding.insert(0, person)
        # face_embedding.insert(1, person_id)

        # create dataframe with embedding to predict
        columns_name = [f'x_{idx}' for idx in range(128)]
        # columns_name.insert(0, 'person name')
        # columns_name.insert(1, 'target')
        df_test = pd.DataFrame(columns=columns_name)
        df_test.loc[len(df_test)] = face_embedding
        print(f'df_test\n{df_test}')
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
    img_idx = np.random.random(len(os.listdir(path)))
    print(img_idx)


# img_test = image_prepocessing('CB_test.jpg')
img_test = image_prepocessing('FK_test.jpg')
embedding = get_embedding(img_test)
print(embedding)
y_pred = model.predict(embedding)
print(y_pred)
print(y_pred[0])
y_pred_probability = model.predict_proba(embedding)
print(y_pred_probability)
person_name = match_person(df_person_ids, y_pred[0])
print(person_name)
get_random_photo(person_name)
