from pydantic import BaseModel
from transformers import pipeline

class NEREvalService(BaseModel):
    def evaluate_NER(self, sentence):
        #Load saved model
        newmodel = pipeline('token-classification', model='NER/saved_model', device=0)
        print('NER results: ', newmodel(sentence))
