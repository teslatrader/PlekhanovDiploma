import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, roc_curve
import tensorflow as tf
from PIL import Image
import face_recognition as fr
import os
import pickle


df_name = 'data_set.pkl'
df = pd.read_pickle(df_name)
print(df)

X = df.drop(['person name', 'target'], axis=1)
y = df['target']
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
# print(X_train, '\n', y_train)
# corr = X_train.corr()
# plt.matshow(corr)
# columns_qty = range(len(corr.columns))
# plt.xticks(columns_qty, corr.columns)
# plt.yticks(columns_qty, corr.columns)
# plt.show()
model = LogisticRegression(multi_class='multinomial')
model.fit(X_train, y_train)
# print(f'X_test\n{X_test}')
y_pred = model.predict(X_test)
f_1 = f1_score(y_test, y_pred, average='micro')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'f1 = {round(f_1, 2)}')
print(f'mse = {round(mse, 2)}')
print(f'mae = {round(mae, 2)}')
print(f'mape = {round(mape, 2)}')

# save trained model
model_name = 'model_logreg'
pickle.dump(model, open(model_name, 'wb'))

