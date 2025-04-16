from pydantic import BaseModel
from transformers import pipeline 

class TranslatorEvalService(BaseModel):
    def evaluate_translator(self, sentence):
        translator = pipeline('translation', model='translator/saved_model', device=0)
        print(translator(sentence))