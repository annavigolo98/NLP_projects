from pydantic import BaseModel
from transformers import pipeline

class NERCustomEvalService(BaseModel):
    def evaluate_custom_ner_dataset(self, string):
        ner = pipeline(
            'token-classification',
            model='NER/saved_model_custom',
            aggregation_strategy='simple',
            device=0
        )

        print(ner(string))