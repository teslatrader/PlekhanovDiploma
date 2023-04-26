import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import face_recognition as fr
import os
import pickle
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
from bing_image_downloader.downloader import download
import shutil
from src.men_to_load import men_to_load
from src.women_to_load import women_to_load


# load the model from disk
model_name = './src/model_logreg'
model = pickle.load(open(model_name, 'rb'))

df_person_ids_name = './src/df_person_id.pkl'
df_person_ids = pd.read_pickle(df_person_ids_name)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.secret_key = 'lkjfpeiojmndsljk=-23j90ifk32-kUJLKdmnf09awejfLJU#$LKJELfjlklf094-3=-=f$oii0$9sj23'

image_name, path_dest = '', '',
list_of_men = men_to_load
list_of_women = women_to_load


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
        img_name = image
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
        os.remove(image)
        return 0
    else:
        os.remove(image)
        return 2


def match_person(df_matches, person_id):
    for per_id in df_matches['person_id']:
        if per_id == person_id:
            return df_matches.loc[per_id]['person']


def get_random_photo(person_name):
    global path_dest
    query = f'face of {person_name}'
    path_to_save = f'./static'
    download(
        query,
        limit=1,
        output_dir=path_to_save,
        adult_filter_off=False,
        timeout=60,
        filter='photo',
        verbose=False
    )

    # adjust name of source folders
    path_query = f'{path_to_save}/{query}'
    path_dest = path_query.replace('face of ', '').replace(' ', '')
    if os.path.exists(path_dest):
        shutil.rmtree(path_dest)
    os.rename(path_query, path_dest)
    img_name = f'{path_dest}/{os.listdir(path_dest)[0]}'
    image_prepocessing(img_name)
    return img_name


def get_prediction(im_name: str):
    img_test = image_prepocessing(im_name)
    embedding = get_embedding(img_test)
    print(f'Embedding of test photo:\n{embedding}')

    if not isinstance(embedding, pd.DataFrame):
        return False, embedding, 0

    prediction = model.predict(embedding)
    print(f'Person id predicted = {prediction}')

    y_pred_probability = model.predict_proba(embedding)[0][prediction[0]]
    print(f'Probability scores by classes:\n{y_pred_probability}')

    person_name = match_person(df_person_ids, prediction[0])
    print(f'Yours highest SUPERSTAR likelihood with {person_name}')
    pred_image = get_random_photo(person_name)
    return person_name, round(y_pred_probability, 2), pred_image


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global image_name, path_dest, list_of_men, list_of_women
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if os.path.exists(image_name):
                os.remove(image_name)
            if os.path.exists(path_dest):
                shutil.rmtree(path_dest)
            image_name = secure_filename(file.filename).split('.')[0] + '.jpg'
            image_name = f'./static/{image_name}'
            file.save(image_name)
            person_name, probability, predicted_image = get_prediction(image_name)
            if not person_name:
                if probability == 0:
                    return redirect('/no_face')
                elif probability == 2:
                    return redirect('/few_faces')
            return render_template('prediction.html', person_name=person_name, probability=probability,
                                   predicted_image=predicted_image, original_image=image_name)
    elif request.method == 'GET':
        return render_template('index.html', list_of_men=list_of_men, list_of_women=list_of_women)


@app.route('/no_face')
def no_faces():
    return render_template('face_not_found.html')


@app.route('/few_faces')
def many_faces():
    return render_template('more_than_one_face.html')


if __name__ == '__main__':
    app.run()
