import sys
import Predict as pr
import datetime
import numpy as np
import os
import configparser

def main():
    try:
        config = configparser.ConfigParser()
        config.read('config.ini')
        predictobj= pr.Prediction(config['Arguments']['dict_path'], config['Arguments']['model_path'], config['Arguments']['emb_path'], config['Arguments']['emb_dim'], config['Arguments']['emb_type'],config['Arguments']['use_emotion'])

        for jsontxt in sys.stdin:
            print(predictobj.predictFromJSONTxt(jsontxt))

    except KeyboardInterrupt:
        sys.stdout.flush()
        pass

if __name__ == "__main__":
    sys.exit(int(main() or 0))