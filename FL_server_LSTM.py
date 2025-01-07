#Server only code using LSTM

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
from _thread import *
import threading
from mlsocket import MLSocket
import time
import matplotlib.pyplot as plt
import pickle

SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 16384

print_lock = threading.Lock()

# Load global data
global_data_path = 'cropser.csv'
global_data = pd.read_csv(global_data_path)
X_global = global_data.iloc[:, 0:7].astype(np.float32).values
y_global = global_data.iloc[:, 7]

lb = LabelEncoder()
y_global = lb.fit_transform(y_global).astype(np.int32)
y_global = to_categorical(y_global)

sc = StandardScaler()
X_global = sc.fit_transform(X_global)
X_global = X_global.reshape((X_global.shape[0], 1, X_global.shape[1]))  # Reshape for LSTM

X_train, X_test, y_train, y_test = train_test_split(X_global, y_global, test_size=0.2, random_state=4)

SERVER_HOST = "10.0.4.102"
SERVER_PORT = 65432
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

local_weights=[]
all_addr=[]

def init_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_global.shape[1], X_global.shape[2])))
    model.add(Dense(y_global.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

t1=time.time()
t=0
num_cl=15
num_round=5

s = MLSocket()
s.bind((SERVER_HOST, SERVER_PORT))

def update_model():
    accuracy=0
    model=init_model()
    model.fit(X_train, y_train, epochs=30, batch_size=32)
    all_addr=[]
    for i in range(num_cl):
        s.listen(2)
        print(f"[*] Listening as {SERVER_HOST}:{SERVER_PORT}")
        print("Waiting for the client to connect... ")
        client_socket, addr=s.accept()
        all_addr.append(client_socket)
        print(f"[+] {addr} is connected.")
    for r in range(num_round):
        local_weights=[]
        i=1
        for conn in all_addr:
            conn.send(model)
            print("Model sent to Client"+str(i))
            i=i+1
        i=1
        for conn in all_addr:
            received = conn.recv(1024)
            print("Model Received from Client"+str(i)) 
            local_weights.append(received)
            i=i+1
        average_weights = [np.mean([model.get_weights()[k] for model in local_weights], axis=0) for k in range(len(local_weights[0].get_weights()))]
        model.set_weights(average_weights)
        history = model.fit(X_train, y_train, epochs=30, batch_size=32)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_test_classes, y_pred)
        CR = classification_report(y_test_classes, y_pred)
        CF=confusion_matrix(y_test_classes, y_pred)
    print("Model is updated finally")
    modelfile='DFLH.sav'
    pickle.dump(model, open(modelfile, 'wb'))
    print("Accuracy of the global model:", accuracy)
    print("Classification Report:", CR)
    print("Confusion matrix:", CF)
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    for conn in all_addr:
        conn.send(model)
        conn.close()
    
update_model()
t2=time.time()
print(t2-t1)

