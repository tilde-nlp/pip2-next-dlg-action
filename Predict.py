import sys
import os

# suppress TensorFlow information messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential, Model
from keras.layers import (Input, LSTM, Dense, Dropout)
from keras.optimizers import Adam, RMSprop
import numpy as np
import pandas as pd
import argparse
import re
import json

from sklearn.model_selection import StratifiedKFold
import random

import datetime 
def ts(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.models import load_model

from DialogFormatReader import encodeToNPwithKnownUniqueVals, readJSONBuffer
from Embeddings import Embeddings


class Prediction:
    def __init__(self, path, modelpath, embpath, embsize, embtype, use_emotion='false'):
        self.uniqueVals = {}
        self.embsize=embsize
        self.embobj= Embeddings(embpath,embsize,embtype)
        self.use_emotion=(use_emotion.lower()=='true')

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line=line.rstrip()
                values = line.split('\t')
                if len(values):
                    self.uniqueVals[values[0]]=values[1:]
        f.close()

        self.model = load_model(modelpath)

        pass

    def predictFromJSONTxt(self, jsontxt):

        responses=[]
        result_line={}
        dialogsVal = readJSONBuffer(jsontxt,True,self.embobj,self.use_emotion)

        xsVal, ysVal = encodeToNPwithKnownUniqueVals(dialogsVal, "action", self.uniqueVals,True,int(self.embsize))
    
        for j, (x_val, y_val) in enumerate(zip(xsVal, ysVal)):
            x_val = x_val.reshape((1,) + x_val.shape)
            result = self.model.predict(x_val)
        
            line=len(result[0])-1
            sorted_index_pos = [index for index, num in sorted(enumerate(result[0][line]), key=lambda x: x[-1], reverse=True)]

            for col in range(len(sorted_index_pos)):
                result_line['action']=self.uniqueVals['action'][sorted_index_pos[col]]
                result_line['probability']=f"{result[0][line][sorted_index_pos[col]]:2.3f}"
                responses.append(result_line)
                result_line={}

        return responses

def main():

    print("Use PredictNextAction.py to predict the next action from the conversation history (in JSON format)")
    pass

if __name__ == "__main__":
    sys.exit(int(main() or 0))

