
from pydantic import BaseModel
from transformers import pipeline
from transformers import AutoTokenizer

class SentimentEvaluationService(BaseModel):

    def evaluate_sentiments(self, sentence):
        newmodel = pipeline('text-classification', model='./saved_model', device=0)
        print('Text classification results: ', newmodel(sentence))