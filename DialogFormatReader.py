import sys
import numpy as np
import pandas as pd

from itertools import chain
np.set_printoptions(edgeitems=30, linewidth=100000)

from Embeddings import Embeddings


def readYamlDocs(path, vectorize=False, embobj=None,use_emotion=False):

    dialogs = []

    from pathlib import Path
    from ruamel.yaml import YAML

    yaml = YAML(typ='safe',pure=True)

    with open(Path(path), 'r',True,'utf-8') as file:
       data = file.read()
       results=yaml.load_all(data)
    file.close()
    #results =yaml.load_all(Path(path)) fails with UnicodeDecodeError for non-latin characters
    n=0
    for dialog in results:
        #n=n+1
        #print(str(n))
        if(dialog==None or 'botid' in dialog[0]):
            continue
        prev_entities = []
        dialog_lines = []
        
        dialog_line0={}
        dialog_line0['entities'] = []
        dialog_line0['prev_entities'] = []
        dialog_line0['intent'] = []
        dialog_line0['actor']='bot'
        dialog_line0['action']='_step0_'
        dialog_line0['valence']=4.0
        dialog_line0['activation']=4.0
        dialog_lines.append(dialog_line0)

        for line in dialog:
            dialog_line={}
            dialog_line['entities'] = []
            dialog_line['prev_entities'] = []
            dialog_line['intent'] = []
            dialog_line['action'] = '-'
            dialog_line['valence']=4.0
            dialog_line['activation']=4.0

            if 'entities' in line:
                dialog_line['entities']=[  str(k) for k, v in line['entities'].items()]

            if 'action' in line:
                dialog_line['action']=line['action']

            if 'actor' in line:
                dialog_line['actor']=line['actor']

            if line['actor']=='user':
                if vectorize==True:
                    if len(line['utterance'].strip(' \n\t'))>0:
                        dialog_line['intent'].extend(embobj.getSentenceVector(line['utterance'].strip(' \n\t')))
                elif 'intents' in line :
                    dialog_line['intent']=line['intents']

            dialog_line['prev_entities'] = prev_entities
            if 'entities' in dialog_line:
                prev_entities=list(set(dialog_line['entities'] + prev_entities))

            if use_emotion:
                if 'valence' in line:
                    dialog_line['valence']=float(line['valence'])
                if 'activation' in line:
                    dialog_line['activation']=float(line['activation'])

            dialog_lines.append(dialog_line)

        if dialog_lines:
            dialogs.append(dialog_lines)


    return dialogs

def readJSONBuffer(jsontxt, vectorize=False,embobj=None,use_emotion=False):

    dialogs = []
    import simplejson as json
    results =json.loads(jsontxt,encoding="utf-8")

    prev_entities = []
    dialog_lines = []
    dialog_line0={}
    dialog_line0['entities'] = []
    dialog_line0['prev_entities'] = []
    dialog_line0['intent'] = []
    dialog_line0['actor']='bot'
    dialog_line0['action']='_step0_'
    dialog_line0['valence']=4.0
    dialog_line0['activation']=4.0
    dialog_lines.append(dialog_line0)

    for line in results:
        dialog_line={}
        dialog_line['entities'] = []
        dialog_line['prev_entities'] = []
        dialog_line['intent'] = []
        dialog_line['action'] = '-'
        dialog_line['valence']=4.0
        dialog_line['activation']=4.0

        if 'entities' in line:
            dialog_line['entities']=[  k for k, v in line['entities'].items()]
        if 'action' in line:
            dialog_line['action']=line['action']

        if 'actor' in line:
            dialog_line['actor']=line['actor']

        if line['actor']=='user':
            if vectorize==True:
                if len(line['utterance'].strip(' \n\t'))>0:
                    dialog_line['intent'].extend(embobj.getSentenceVector(line['utterance'].strip(' \n\t')))
            elif 'intents' in line :
                dialog_line['intent']=line['intents']

        dialog_line['prev_entities'] = prev_entities
        if 'entities' in dialog_line:
            prev_entities=list(set(dialog_line['entities'] + prev_entities))

        if use_emotion:
            if 'valence' in line:
                dialog_line['valence']=float(line['valence'])
            if 'activation' in line:
                dialog_line['activation']=float(line['activation'])

        dialog_lines.append(dialog_line)

    dialogs.append(dialog_lines)

    return dialogs

def encodeToNP(dialogs, predCol, vectorize=False, embsize=300):
    dfs = [pd.DataFrame(d) for d in dialogs]
    for d in dfs:
        d.values[d.values==None] = "-"

    df = pd.concat(dfs)
    isList = {}
    uniqueVals = {}
    cols = list(df)
    totalOffset = 0
    offsets = [0] * len(list(df))

    for i, col in enumerate(cols):
        if col == 'valence' or col == 'activation':
            offsets[i]=totalOffset
            totalOffset +=1
        elif vectorize==False or (vectorize==True and col != 'intent'):
            if (df[col].apply(type) == list).any():
                isList[col] = True
                uniqueVals[col] = np.array(sorted(list(set(df[col].sum()))), dtype="object")
            else:
                isList[col] = False
                vals = df[col].unique()
                vals = vals[vals != np.array(None)]
                vals.sort()
                uniqueVals[col] = vals
            offsets[i] = totalOffset 
            totalOffset += len(uniqueVals[col])
        else:
            offsets[i]=totalOffset

    retval = []
    ys = []
    totalOffsetSentEmbedd=totalOffset

    if vectorize:
        totalOffsetSentEmbedd=totalOffsetSentEmbedd+embsize

    for d in dfs:
        arr = np.zeros(shape=(len(d), totalOffsetSentEmbedd + 1)) # add first dialog marker
        y = np.zeros(shape=(len(d), len(uniqueVals[predCol])))

        for line in range(len(d)):
            for i, col in enumerate(cols):
                if col == 'valence' or col == 'activation': #columns that are not one-hot
                    arr[line, offsets[i]]=d.loc[line, col]
                elif vectorize==True and col == 'intent':
                    idx=0
                    for val in d.loc[line, col]:
                        arr[line, totalOffset+idx]=val
                        idx += 1
                else:
                    if isList[col]:
                        for val in d.loc[line, col]:
                            arr[line, offsets[i] + np.where(uniqueVals[col] == val)[0][0]] = 1
                    else:
                        arr[line, offsets[i] + np.where(uniqueVals[col] == d.loc[line, col])[0][0]] = 1

            if line < len(d)-1:
                y[line, np.where(uniqueVals[predCol] == d.loc[line+1, predCol])[0][0]] = 1
            else:
                y[line, np.where(uniqueVals[predCol] == "-")[0][0]] = 1
        arr[0, totalOffset] = 1
        retval.append(arr)
        ys.append(y)

    return retval, ys, uniqueVals

def encodeToNPwithKnownUniqueVals(dialogs, predCol, uniqueVals, vectorize=False, embsize=300):

    dfs = [pd.DataFrame(d) for d in dialogs]
    for d in dfs:
        d.values[d.values == None] = "-"
    df = pd.concat(dfs)
    isList = {}
    cols = list(df)
    totalOffset = 0
    offsets = [0] * len(list(df))
    for i, col in enumerate(cols):
        if col == 'valence' or col == 'activation':
            offsets[i]=totalOffset
            totalOffset +=1
        elif vectorize==False or (vectorize==True and col != 'intent'):
            if (df[col].apply(type) == list).any():
                isList[col] = True
            else:
                isList[col] = False
            offsets[i] = totalOffset 
            totalOffset += len(uniqueVals[col])
        else:
            offsets[i]=totalOffset

    retval = []
    ys = []

    totalOffsetSentEmbedd=totalOffset
    if vectorize:
        totalOffsetSentEmbedd=totalOffsetSentEmbedd+embsize

    for d in dfs:
        arr = np.zeros(shape=(len(d), totalOffsetSentEmbedd + 1)) # add first dialog marker
        y = np.zeros(shape=(len(d), len(uniqueVals[predCol])))
        for line in range(len(d)):
            for i, col in enumerate(cols):
                if col == 'valence' or col == 'activation': #columns that are not one-hot
                    arr[line, offsets[i]]=d.loc[line, col]
                elif vectorize==True and col == 'intent':
                    idx=0
                    for val in d.loc[line, col]:
                        arr[line, totalOffset+idx]=val
                        idx += 1
                else:
                    if isList[col]:
                        for val in d.loc[line, col]:
                            for idx,valunique in enumerate(uniqueVals[col]):
                                if valunique == val:
                                    arr[line, offsets[i] + idx] = 1
                    else:
                        for idx,valunique in enumerate(uniqueVals[col]):
                            if valunique == d.loc[line, col]:
                                arr[line, offsets[i] + idx] = 1
            if line < len(d)-1:
                for idx,valunique in enumerate(uniqueVals[predCol]):
                    if valunique == d.loc[line+1, predCol]:
                        y[line, idx] = 1
            else:
                for idx,valunique in enumerate(uniqueVals[predCol]):
                    if valunique == "-":
                        y[line, idx] = 1
        arr[0, totalOffset] = 1
        retval.append(arr)
        ys.append(y)


    return retval, ys


def main():

    pass

if __name__ == "__main__":
    sys.exit(int(main() or 0))
