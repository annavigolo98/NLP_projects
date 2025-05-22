import evaluate 
import numpy as np

class MetricEvaluator:
    def __init__(self, tokenizer, translated_language: str):
        self.tokenizer = tokenizer
        self.metric_bleu = evaluate.load('sacrebleu')
        #self.metric_bert = evaluate.load('bertscore')
        self.translated_language = translated_language

    def __call__(self, decoded_predictions, decoded_labels):
        bleu_score = self.metric_bleu.compute(predictions=decoded_predictions, 
                                              references=decoded_labels)
        
        return bleu_score['score']