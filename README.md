# pip2-next-dlg-action
Next Dialog Action Prediction model

## Included
- Scripts for training and running the next dialog action prediction model `TrainModel.py`, `Embeddings.py`, `DialogFormatReader.py`, `PredictNextAction.py`, `Predict.py`
- Configuration files `train_config.ini`, `config.ini`

## Prerequisites
- Scripts tested with `Python 3.7`
- Scripts are using the following Python libraries: `keras`, `pandas`, `numpy`, `nltk`, `fasttext`, `sklearn`

## Training model
- Training is invoked by running command `python TrainModel.py`
- File `train_config.ini` contains:
-	`dict_path` - path for the model dictionary,
-	`model_path` - path for the trained model,
-	`training_data_dir` - directory of training data (all files in the directory will be used for model training),
-	`emb_type` - embedding engine (`fasttext`, `bert` or `transformer`), 
-	`emb_path` - path of the embedding file for the fasttext model or name of the transformer model (e.g., `LaBSE`),
-	`emb_dim` - embedding dimensions,
-	`other_data_dir` - directory of training data in different language,
-	`other_emb_type` - embedding engine for data in different language(`fasttext`, `bert` or `transformer`), 
-	`other_emb_path` - path of the embedding file for data in different language for the fasttext model or name of the transformer model (e.g., `LaBSE`),
-	`test_on_all_sets` - values True/False, if data in both languages should be used for testing
-	`train_on_all_sets` - values True/False, if data in both languages should be used for training
-	`use_emotion` - use of emotion variables in model training,
-	`epochs` - number of epochs to train the model,
-	`xvalidation_folds` - number of folds for the cross validation.
- Training data are files in YAML format.
Example:
<pre>
- action: "starts"
  actor: "bot"
- actor: "user"
  valence: "4"
  activation: "4"
  utterance: "hello"
  input_mode: "txt"
  input_language: "LV"
</pre>
## Using model
- Trained model is run by command `python PredictNextAction.py`. It returns the list of actions and probabilities in JSON format.
- File `config.ini` must contain path of the dictionary, path of the model, path of the embedding file for fasttext model or name of transformer model, embedding dimensions, embedding engine (`fasttext`, `bert` or `transformer`), use of emotion variables in model training.
- Input in JSON format contain conversation history.
Example:

`[{"action":"start","actor":"bot"},{"actor":"user","valence":"4","activation":"4","entities":{},"utterance":"hello"}]`

- Output in JSON format contain possible next actions.
Example:

`[{"action": "greeting", "probability": "0.829"}, {"action": "didnotunderstand", "probability": "0.034"}, ... ]`

## Acknowledgment
The research has been supported by the European Regional Development Fund within the project “Multilingual Artificial Intelligence Based Human Computer Interaction” No. 1.1.1.1/18/A/148.

## Licence
Licensed under a [MIT](https://opensource.org/licenses/MIT)