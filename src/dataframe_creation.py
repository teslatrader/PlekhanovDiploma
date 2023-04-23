import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import face_recognition as fr
import os

# setup path variables
path_inputs = '../inputs'
persons = os.listdir(path_inputs)

# create empty dataframe with target and features columns names
df_name = 'data_set.pkl'
columns_name = [f'x_{idx}' for idx in range(128)]
columns_name.insert(0, 'target')
df = pd.DataFrame(columns=columns_name)

# filling dataframe via iteration through each folder with images
for person in persons:
    persons_path = f'{path_inputs}/{person}'
    imgs = os.listdir(persons_path)
    for image in imgs:
        img = fr.load_image_file(f'{persons_path}/{image}')
        face_location = fr.face_locations(img, number_of_times_to_upsample=1)
        if len(face_location) == 1:
            # creating face embedding and linking it with target person
            face_embedding = fr.face_encodings(img)[0]
            face_embedding = face_embedding.tolist()
            face_embedding.insert(0, person)

            # append embedding to dataframe
            df.loc[len(df)] = face_embedding

            # visualization of face frame (just for check if it works)
            # fl = face_location[0]
            # y1 = fl[0]
            # y2 = fl[2]
            # x1 = fl[1]
            # x2 = fl[3]
            # xy = (x2, y1)
            # width = x1 - x2
            # height = y2 - y1
            # ax = plt.gca()
            # frame = patches.Rectangle(xy, width, height, linewidth=1, edgecolor='Magenta', facecolor='none')
            # ax.add_patch(frame)
            # plt.imshow(img)
            # plt.show()
            # print(f'\
            #     Face location coordinates: {fl}\n\
            #     Start point coordinates: {xy}\n\
            #     Width: {width}\n\
            #     Height: {height}\
            #     ')

df.to_pickle(df_name)
# print(df)
# print(df.shape)
