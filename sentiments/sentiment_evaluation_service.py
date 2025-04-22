
from pydantic import BaseModel
from transformers import pipeline

class SentimentEvaluationService(BaseModel):

    def evaluate_sentiments(self, sentence):
        newmodel = pipeline('text-classification', model='sentiments/saved_model', device=0)
        print('Text classification results: ', newmodel(sentence))