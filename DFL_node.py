#DFLcode with four nodes

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd
import socket
import tqdm
import json
import os
from _thread import *
import threading
from mlsocket import MLSocket
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


SERVER_HOST = "10.0.4.102"
SERVER_PORT = 65432

local_data = pd.read_csv('crop.csv')
X_local = local_data.iloc[:, 0:7].astype(np.float32).values
y_local = local_data.iloc[:, 7]

lb = LabelEncoder()
y_local = lb.fit_transform(y_local).astype(np.int32)
y_local = to_categorical(y_local)

sc = StandardScaler()
X_local = sc.fit_transform(X_local)
X_local = X_local.reshape((X_local.shape[0], 1, X_local.shape[1]))  # Reshape for LSTM

X_train, X_test, y_train, y_test = train_test_split(X_local, y_local, test_size=0.2, random_state=41)

t=0
num_peer=3


def initmodel():
    with MLSocket() as s1:
        s1.connect((SERVER_HOST, SERVER_PORT)) # Connect to the port and host
        model=s1.recv(1024)
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
        s1.close()
    return model


OHOST = "10.0.4.104"
OPORT = 65002

num_round=3
j=0


accuracy=0

def connectasserver(model):
    with MLSocket() as s:
        s.bind((OHOST, OPORT))
        s.listen()
        print("waiting.....")
        peer_models=[]
        all_addr=[]
        for i in range(num_peer):
            t1=time.time()
            conn, addr= s.accept()
            all_addr.append(conn)
        for conn in all_addr:
            received=conn.recv(1024)
            peer_models.append(received)
            print("Model received")
        average_weights = [np.mean([model.get_weights()[k] for model in peer_models], axis=0) for k in range(len(peer_models[0].get_weights()))]
        model.set_weights(average_weights)
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
        for conn in all_addr:
            conn.send(model)
            print("Model sent")
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_test_classes, y_pred)
        print("Accuracy of the peer model:", accuracy)
        CR = classification_report(y_test_classes, y_pred)
        CM = confusion_matrix(y_test_classes, y_pred)
        print(CR)
        print(CM)
        RMSE=sqrt(mean_squared_error(y_test_classes, y_pred))
        print(RMSE)
    return model, all_addr


def connectasclient(model, SHOST, SPORT):
    with MLSocket() as s2:
        s2.connect((SHOST, SPORT)) # Connect to the port and host
        s2.send(model)
        model=s2.recv(1024)
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
        s2.close()
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_test_classes, y_pred)
        print("Accuracy of the peer model:", accuracy)
        CR = classification_report(y_test_classes, y_pred)
        CM = confusion_matrix(y_test_classes, y_pred)
        print(CR)
        print(CM)
        RMSE=sqrt(mean_squared_error(y_test_classes, y_pred))
        print(RMSE)
    return model


def modelupdate():
    model=initmodel()
    for i in range(num_round):
        time.sleep(50)
        model=connectasclient(model,"10.0.4.112", 65001)
        model, all_addr=connectasserver(model)
        time.sleep(50)
        model=connectasclient(model,"10.0.4.115", 65006)
        time.sleep(50)
        model=connectasclient(model,"10.0.4.105", 65007)
    for conn in all_addr:
        conn.send(model)
        conn.close()

t1=time.time()
modelupdate()
t2=time.time()
print(t2-t1)
    
