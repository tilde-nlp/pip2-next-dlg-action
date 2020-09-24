# pip2-next-dlg-action
Next Dialog Action Prediction model

## Included
- Scripts for training and running the next dialog action prediction model 'TrainModel.py', 'Embeddings.py', 'DialogFormatReader.py', 'PredictNextAction.py', 'Predict.py'
- Configuration files 'train_config.ini', 'config.ini'

## Prerequisites
- Scripts tested with 'Python 3.7'
- Scripts are using the following Python libraries: 'keras', 'pandas', 'numpy', 'nltk', 'fasttext', 'sklearn'

## Training model
- Training is invoked by running command 'python TrainModel.py'
- File 'train_config.ini' contains path for the model dictionary, path for the model, path of the embedding file, directory of training data (all files in the directory will be used for model training), embedding dimensions, embedding engine ('fasttext' is the only option), if the model will be trained using emotion variables, number of epochs, number of folds for the cross validation.
- Training data are files in YAML format:
'
- action: "starts"
  actor: "bot"
- actor: "user"
  valence: "4"
  activation: "4"
  utterance: "ko tu māki?"'

## Using model
- Trained model is run by command 'python PredictNextAction.py'. It returns the list of actions and probabilities in JSON format.
- File 'config.ini' must contain path of the dictionary, path of the model, path of the embedding file, embedding dimensions, embedding engine ('fasttext' is the only option), if the model is trained using emotion variables.
- Input in JSON format containg conversation history:
'[{"action":"start","actor":"bot"},{"actor":"user","valence":"4","activation":"4","entities":{},"utterance":"hello"}]'
- Output in JSON format containing possible next actions:
'[{"action": "greeting", "probability": "0.829"}, {"action": "didnotunderstand", "probability": "0.034"}, ... ]'

## Acknowledgment
The research has been supported by the European Regional Development Fund within the project “Multilingual Artificial Intelligence Based Human Computer Interaction” No. 1.1.1.1/18/A/148.

## Licence
The models are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).