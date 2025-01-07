#Client only code

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.utils import to_categorical
import pandas as pd
import socket
import tqdm
import json
import os
import threading
from mlsocket import MLSocket
import time
import pickle

HOST = "10.0.4.102"
PORT = 65432
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

t1=time.time()
local_data = pd.read_csv('crop1.csv')

X_local = local_data.iloc[:,0:7].astype(np.float32).values
y_local = local_data.iloc[:, 7]

lb = LabelEncoder()
y_local = lb.fit_transform(y_local).astype(np.int32)
y_local = to_categorical(y_local)

sc = StandardScaler()
X_local = sc.fit_transform(X_local)
X_local = X_local.reshape((X_local.shape[0], 1, X_local.shape[1]))  # Reshape for LSTM

X_train, X_test, y_train, y_test = train_test_split(X_local, y_local, test_size=0.2, random_state=4)



def modelupdate():
    t=0
    with MLSocket() as s:
        s.connect((HOST, PORT)) # Connect to the port and host
        while True:
            model=s.recv(1024)
            print("Received model")
            model.fit(X_train, y_train, epochs=30, batch_size=32)
            s.send(model)
            y_pred = np.argmax(model.predict(X_test), axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            accuracy = accuracy_score(y_test_classes, y_pred)
            t2=time.time()
            CR = classification_report(y_test_classes, y_pred)
            CM = confusion_matrix(y_test_classes, y_pred)
            print("Accuracy of the local model:", accuracy)
            print(CM)
            print(CR)
            t2=time.time()
            t=t+(t2-t1)
            print(t)
    return model

finalmodel=modelupdate()







