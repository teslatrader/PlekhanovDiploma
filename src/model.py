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

# save trained model
model_name = 'model_logreg'
pickle.dump(model, open(model_name, 'wb'))

