from pydantic import BaseModel 
import evaluate 
import numpy as np

class MetricEvaluator(BaseModel):
    def compute_metrics(self, logits_and_labels):
            #This metrics takes two inputs:
            #predictions: a lisi of lists of string predictions: the predicted NER strings 
            #references: a list of lists of string references: the NER labels
            #returns: f1_score, recall, precision, accuracy
            metric = evaluate.load('seqeval')
            label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
 
            logits, labels = logits_and_labels
            preds=np.argmax(logits, axis=-1)


            str_labels=[
                [label_names[t] for t in label if t!=-100] for label in labels
            ]
            
            str_preds = [
                [label_names[p] for p, t in zip(pred, targ) if t!=-100] for pred, targ in zip(preds, labels)
            ]
            metric_evaluation = metric.compute(predictions=str_preds, references=str_labels)
            return {'precision': metric_evaluation['overall_precision'],
                    'recall': metric_evaluation['overall_recall'],
                    'f1': metric_evaluation['overall_f1'],
                    'accuracy': metric_evaluation['overall_accuracy']}