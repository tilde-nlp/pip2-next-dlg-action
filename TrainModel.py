import sys
import statistics
import os

# suppress TensorFlow information messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import (Input, LSTM, Dense, Dropout)
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l1

import numpy as np
import pandas as pd
import argparse
import re
import json
import configparser
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import random

import datetime 
def ts(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.models import load_model

from DialogFormatReader import readYamlDocs, encodeToNP
from Embeddings import Embeddings

np.set_printoptions(formatter={'float': lambda x: f"{x:.5f}"})

def named_logs(model, logs, pref=""):
  result = {}
  for l in zip(model.metrics_names, logs):
    result[pref + l[0]] = l[1]
  return result

def createModel(input_size, output_size):

    input = Input(shape=(None, input_size))
    lstm = LSTM(100, return_sequences=True, dropout=0.2, activity_regularizer=l1(0.001))(input)
    drpout = Dropout(0.1)(lstm) #neļauj val_loss sākt augt tik agri kā bez šī slāņa
    output = Dense(output_size, activation='softmax')(drpout)

    model = Model(input, output)
    return model

def doTrainVal(xsAll, ysAll, trainsamples, modelpath, epochs=400):
    input_size, output_size = len(xsAll[0][0]), len(ysAll[0][0])

    xsTrain=xsAll[:trainsamples]
    ysTrain=ysAll[:trainsamples]

    xsVal=xsAll[trainsamples:]
    ysVal=ysAll[trainsamples:]

    startTime = datetime.datetime.now()
    dateText = startTime.strftime("%Y-%m-%d")
    timeText = startTime.strftime("%H-%M-%S")
        
    model = createModel(input_size, output_size)
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    logDir = f"./tbLog/{dateText}/{timeText}/xval"
    tensorboard = TensorBoard(log_dir=logDir, histogram_freq=0, batch_size=1, write_graph=True, write_grads=True)
    tensorboard.set_model(model) 

    best_accuracy=0.0
    prev_accuracy=0.0
    min_delta=0.0001
    patience=3
    test_loss_delta=0.05
    not_improving=patience
    
    for epoch in range(epochs):
        print(f"{ts()}: Epoch {epoch+1}/{epochs}: ", end='')
        epochScores = np.zeros((2))
        testScores = np.zeros((2))

        for i, (x, y) in enumerate(zip(xsTrain, ysTrain)):
            x = x.reshape((1,) + x.shape)
            y = y.reshape((1,) + y.shape)
            currscores = model.train_on_batch(x, y)
            epochScores += np.array(currscores)
        epochScores /= len(xsTrain)
        print(epochScores, end=' ')

        for j, (x_val, y_val) in enumerate(zip(xsVal, ysVal)):
            x_val = x_val.reshape((1,) + x_val.shape)
            y_val = y_val.reshape((1,) + y_val.shape)
            currscores = model.test_on_batch(x_val, y_val)
            testScores += np.array(currscores)
        testScores /= len(xsVal)
        print(testScores)

        tensorboard.on_epoch_end(epoch, {**named_logs(model, epochScores), **named_logs(model, testScores, "val_")})
            
        if epochScores[1] > best_accuracy:
            model.save(modelpath)
            best_accuracy=epochScores[1]

        if abs(prev_accuracy - epochScores[1]) < min_delta:
            not_improving=not_improving-1
        else:
            not_improving=patience

        if not_improving<0:
            break

        prev_accuracy=epochScores[1]

    tensorboard.on_train_end(None)

    model.save(modelpath)

    return None

def doNXVal(xsAll, ysAll, bestmodel, epochs=400, splits=10):
    input_size, output_size = len(xsAll[0][0]), len(ysAll[0][0])

    kf = KFold(n_splits = splits, shuffle = True)

    min_delta=0.001
    patience=3
    test_loss_delta=0.05
    
    scores = []
    iterscores=[]

    for i in range(splits):
    
        model = createModel(input_size, output_size)
        opt = Adam()
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

        startTime = datetime.datetime.now()
        dateText = startTime.strftime("%Y-%m-%d")
        timeText = startTime.strftime("%H-%M-%S")
        logDir = f"tbLog\\{dateText}\\{timeText}\\xval"
        PATH = os.path.join('.', logDir)
        tensorboard = TensorBoard(log_dir=PATH, histogram_freq=0, write_graph=True)
        tensorboard.set_model(model)

        best_accuracy=0.0
        saved_model_test_accuracy=0.0
        prev_accuracy=0.0
        min_test_loss=10.0
        not_improving=patience
    
        result = next(kf.split(xsAll), None)
        xsTrain = [xsAll[i] for i in result[0]]
        ysTrain = [ysAll[i] for i in result[0]]
        xsVal = [xsAll[i] for i in result[1]]
        ysVal = [ysAll[i] for i in result[1]]
        
        scores=[]

        for epoch in range(epochs):
            print(f"{ts()}: Epoch {epoch+1}/{epochs}: ", end='')
            epochScores = np.zeros((2))
            testScores = np.zeros((2))

            for i, (x, y) in enumerate(zip(xsTrain, ysTrain)):
                x = x.reshape((1,) + x.shape)
                y = y.reshape((1,) + y.shape)
                currscores = model.train_on_batch(x, y)
                epochScores += np.array(currscores)
            epochScores /= len(xsTrain)
            print(epochScores, end=' ')

            for j, (x_val, y_val) in enumerate(zip(xsVal, ysVal)):
                x_val = x_val.reshape((1,) + x_val.shape)
                y_val = y_val.reshape((1,) + y_val.shape)
                currscores = model.test_on_batch(x_val, y_val)
                testScores += np.array(currscores)
            testScores /= len(xsVal)
            print(testScores)
            tensorboard.on_epoch_end(epoch, {**named_logs(model, epochScores), **named_logs(model, testScores, "val_")})
            
            if epochScores[1] > best_accuracy:
                if os.path.isfile(bestmodel):
                    os.remove(bestmodel)
                model.save(bestmodel)
                best_accuracy=epochScores[1]
                saved_model_test_accuracy=testScores[1]

            if abs(prev_accuracy - epochScores[1]) < min_delta:
                not_improving=not_improving-1
            else:
                not_improving=patience

            if not_improving<0:
                print('Early stopping, accuracy is not increasing.\n')
                break

            if min_test_loss > testScores[0]:
                min_test_loss=testScores[0]
            elif testScores[0] - min_test_loss > test_loss_delta: #testLoss increases, overfitting
                print('Early stopping, test loss increases.\n')
                break

            prev_accuracy=epochScores[1]

        tensorboard.on_train_end(None)
        iterscores.append(saved_model_test_accuracy) #test accuracy of saved model (best trained model)
        #iterscores.append(np.mean(scores)) - mean test accuracy of all epochs
        #scores.clear - this function did not clear the array

    print('Scores from each Iteration: ', iterscores)
    print('Average K-Fold Score:' , np.mean(iterscores))
    res = statistics.pstdev(iterscores) 
    print("Standard deviation of sample is: " + str(res))
    print("Number of dialogs is: " + str(len(xsAll))) 

    return None

def main():
    try:
        # $ python Trainmodel.py [botid]
        # config['Arguments']['dict_path'] and  config['Arguments']['model_path'] concatenated with botid if it is specified

        ininame='train_config.ini'
        botid=''
        if len(sys.argv)>1:
            botid = sys.argv[1]
        config = configparser.ConfigParser()
        config.read(ininame)
 
        embobj= Embeddings(config['Arguments']['emb_path'], config['Arguments']['emb_dim'], config['Arguments']['emb_type'])
        
        dialogs4Training = []
        for r, d, f in os.walk(config['Arguments']['training_data_dir']):
            for filepath in f:
                dialogs4Training = dialogs4Training + readYamlDocs(os.path.join(r,filepath),True,embobj,config['Arguments']['use_emotion'].lower()=='true')

        if len(dialogs4Training)==0:
            print("No dialogs to train!!!")
            sys.exit()

        xsAll, ysAll, uniqueVals = encodeToNP(dialogs4Training, "action",True,embobj.embsize)

        with open(config['Arguments']['dict_path']+botid, 'w', encoding='utf-8') as f:
            for k, v in uniqueVals.items():
                f.write('\n' + k)
                for val in v:
                    f.write('\t' + val)
        f.close()

        xfolds=int(config['Arguments']['xvalidation_folds'])
        if(xfolds>0):
            doNXVal(xsAll, ysAll, config['Arguments']['model_path']+botid, epochs=int(config['Arguments']['epochs']), splits=xfolds)
        else:
            trainsamples = max(len(xsAll)-1,int(len(xsAll) - len(xsAll)/10)) #number of training samples, 1/10 for validation
            doTrainVal(xsAll, ysAll, trainsamples, config['Arguments']['model_path']+botid, epochs=int(config['Arguments']['epochs']))
        pass

    except KeyboardInterrupt:
        sys.stdout.flush()
        pass

if __name__ == "__main__":
    sys.exit(int(main() or 0))