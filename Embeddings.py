import sys
import os

import numpy as np

# suppress TensorFlow information messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from nltk.tokenize import sent_tokenize, word_tokenize
import fasttext
from bert_serving.client import BertClient
from sentence_transformers import SentenceTransformer

class Embeddings:
	def __init__(self, path, embsize, embtype):
		self.embsize=int(embsize)
		self.embtype=embtype
		self.model=None

		if (embtype=='bert'):
			self.model = BertClient(ip=path)
		elif (embtype=='transformer'):
			self.model = SentenceTransformer(path)
		elif (embtype=='fasttext') and os.path.exists(path):
			self.model = fasttext.load_model(path)
			self.embsize=self.model.get_dimension()

		pass


	def getSentenceVector(self, sentence):
		sentvec=np.zeros(self.embsize, dtype = float)

		if (self.model and len(sentence)>0):
			if (self.embtype=='fasttext'):
				sentvec=self.model.get_sentence_vector(' '.join(word_tokenize(sentence.lower())))
			elif  (self.embtype=='bert'):
				sentvec=self.model.encode([' '.join(word_tokenize(sentence.lower()))])[0]
			elif  (self.embtype=='transformer'):
				sentvec=self.model.encode([' '.join(word_tokenize(sentence.lower()))])[0]
		return sentvec

def main():


	pass

if __name__ == "__main__":
	sys.exit(int(main() or 0))

