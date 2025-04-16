import evaluate 
import numpy as np

class MetricEvaluator:
    def __init__(self, tokenizer, translated_language: str):
        self.tokenizer = tokenizer
        self.metric_bleu = evaluate.load('sacrebleu')
        self.metric_bert = evaluate.load('bertscore')
        self.translated_language = translated_language

    def __call__(self, preds_and_labels):
        preds, labels = preds_and_labels
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        #labels = np.where(labels == -100, labels, self.tokenizer.pad_token_id)
        labels = np.where(labels == -100, self.tokenizer.pad_token_id, labels)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        bleu_score = self.metric_bleu.compute(predictions=decoded_preds, 
                                              references=decoded_labels)
        
        bert_score = self.metric_bert.compute(predictions=decoded_preds, 
                                              references=decoded_labels, 
                                              lang=self.translated_language)
        return {'bleu': bleu_score, 'bert_score': np.mean(bert_score['f1'])}