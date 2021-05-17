import sys
import statistics
import os
import fnmatch
# suppress TensorFlow information messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import collections
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import (Input, LSTM, Dense, Dropout, GaussianNoise, GaussianDropout,AlphaDropout,BatchNormalization)
from keras.optimizers import Adam, RMSprop, SGD
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
import pydot
from keras.utils import plot_model
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
    drpout2 = Dropout(0.1)(lstm)
    output = Dense(output_size, activation='softmax')(drpout2)

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
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
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

def evaluateActions(xs, ys, bestmodel, dictpath):

    responses = []
    uniqueVals =  {}
    with open(dictpath, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line):
                line=line.rstrip()
                values = line.split('\t')
                if len(values)>1:
                    uniqueVals[values[0]]=values[1:]
    f.close()

    model = load_model(bestmodel)

    for j, (x_val, y_val) in enumerate(zip(xs, ys)):
        x_val = x_val.reshape((1,) + x_val.shape)
        result = model.predict(x_val)
        
        for line in range(0,len(result[0])-1):
            sorted_index_pos = [index for index, num in sorted(enumerate(result[0][line]), key=lambda x: x[-1], reverse=True)]
            ypos=np.argmax(y_val[line])
            i=0
            for col in range(len(sorted_index_pos)):
                i=i+1
                result_line={}
                if ypos == float(sorted_index_pos[col]):
                    responses.append([uniqueVals['action'][sorted_index_pos[col]],f"{result[0][line][sorted_index_pos[col]]:2.3f}", i])
                    break
    resdf = pd.DataFrame(responses, columns = ['Action', 'Confidence','Range'])
    uniqueactions=resdf['Action'].unique()

    with open(bestmodel+'.txt', 'a', encoding='utf-8') as f:
        print('\nAction\tCount\tConfidence\tRange mean\tRange mode', file=f)
        for action in uniqueactions:
            singleaction=resdf[resdf['Action']==action]
            c = collections.Counter(singleaction['Range'])
            mode_val = [k for k, v in c.items() if v == c.most_common(1)[0][1]]
            print(action , len(singleaction.index),f"{singleaction['Confidence'].astype(float).mean():2.4f}",f"{singleaction['Range'].astype(float).mean():2.3f}",', '.join(map(str, mode_val)),sep='\t', file=f)
        pd.set_option('display.max_rows', None)
        print(resdf,file=f)
    f.close()

    return None

def doNXVal(xsAll, ysAll, xsOther,ysOther,train_on_all_sets,test_on_all_sets,bestmodel, dictpath, epochs=400, splits=10):
    input_size, output_size = len(xsAll[0][0]), len(ysAll[0][0])
    numdlgs=len(xsAll)

    if len(ysOther)==0:
        otheroutput_size = 0
    else:
        otheroutput_size = len(ysOther[0][0])
    if len(xsOther)==0:
        otherinput_size = 0
    else:
        otherinput_size = len(xsOther[0][0])

    kf = KFold(n_splits = splits, shuffle = True)

    min_delta=0.001
    patience=3
    test_loss_delta=0.1
    
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

        if(input_size==otherinput_size and train_on_all_sets):
            xsTrain = [xsAll[i] for i in result[0]] + [xsOther[i] for i in result[0]]
            ysTrain = [ysAll[i] for i in result[0]] + [ysAll[i] for i in result[0]]
        else:
            xsTrain = [xsAll[i] for i in result[0]]
            ysTrain = [ysAll[i] for i in result[0]]

        if(input_size==otherinput_size):
            if(test_on_all_sets):#validating with both sets
                xsVal = [xsAll[i] for i in result[1]] + [xsOther[i] for i in result[1]]
                ysVal = [ysAll[i] for i in result[1]] + [ysAll[i] for i in result[1]]
            else:#validating with the parallel set
                xsVal = [xsOther[i] for i in result[1]]
                ysVal = [ysOther[i] for i in result[1]]
        else:#validating with the same set
            xsVal = [xsAll[i] for i in result[1]]
            ysVal = [ysAll[i] for i in result[1]]
        
        scores=[]
        print(f"Split nr: {i}")
        print(f"Num validation dialogs: {len(xsVal)}")
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
        
        evaluateActions(xsVal, ysVal, bestmodel, dictpath)

        iterscores.append(saved_model_test_accuracy) #test accuracy of saved model (best trained model)
        #iterscores.append(np.mean(scores)) - mean test accuracy of all epochs
        #scores.clear - this function did not clear the array
    if(input_size==otherinput_size):
        evaluateActions(xsAll+xsOther, ysAll+ysAll, bestmodel, dictpath)
        numdlgs = numdlgs + len(xsOther)
    else:
        evaluateActions(xsAll, ysAll, bestmodel, dictpath)

    with open(bestmodel+'.txt', 'a', encoding='utf-8') as f:
        print("Number of dialogs is: " + str(numdlgs),file=f) 
        print('Scores from each Iteration: ', iterscores,file=f)
        print('Average K-Fold Score:' , np.mean(iterscores),file=f)
        res = statistics.pstdev(iterscores) 
        print("Standard deviation of sample is: " + str(res),file=f)
        f.close()

    return None

def main():
    try:
        # $ python Trainmodel.py [botid]
        # config['Arguments']['dict_path'] and  config['Arguments']['model_path'] concatenated with botid if it is specified

        ininame='train_config.ini'
        botid='2'
        if len(sys.argv)>1:
            botid = sys.argv[1]
        config = configparser.ConfigParser()
        config.read(ininame)
 
        embobj= Embeddings(config['Arguments']['emb_path'], config['Arguments']['emb_dim'], config['Arguments']['emb_type'])

        dialogs4Training = []
        for r, d, f in os.walk(config['Arguments']['training_data_dir']):
           for filepath in fnmatch.filter(f, "*.yaml"):
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
            xsOther=[]
            ysOther=[]
            if (len(config['Arguments']['other_data_dir'])>0):
                if(config['Arguments']['emb_path'] != config['Arguments']['other_emb_path']):
                    embobj= Embeddings(config['Arguments']['other_emb_path'], config['Arguments']['emb_dim'], config['Arguments']['other_emb_type'])
                otherdialogs4Training = []
                for r, d, f in os.walk(config['Arguments']['other_data_dir']):
                    for filepath in fnmatch.filter(f, "*.yaml"):
                        otherdialogs4Training = otherdialogs4Training + readYamlDocs(os.path.join(r,filepath),True,embobj,config['Arguments']['use_emotion'].lower()=='true')
                xsOther, ysOther, uniqueVals = encodeToNP(otherdialogs4Training, "action",True,embobj.embsize)

            doNXVal(xsAll, ysAll, xsOther, ysOther, config['Arguments']['train_on_all_sets'].lower()=='true', config['Arguments']['test_on_all_sets'].lower()=='true',config['Arguments']['model_path']+botid, config['Arguments']['dict_path']+botid, epochs=int(config['Arguments']['epochs']), splits=xfolds)
        else:
            trainsamples = max(len(xsAll)-1,int(len(xsAll) - len(xsAll)/10)) #number of training samples, 1/10 for validation
            doTrainVal(xsAll, ysAll, trainsamples, config['Arguments']['model_path']+botid, epochs=int(config['Arguments']['epochs']))
        pass

    except KeyboardInterrupt:
        sys.stdout.flush()
        pass

if __name__ == "__main__":
    sys.exit(int(main() or 0))