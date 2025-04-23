import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class MetricEvaluatorCustom:

    def __init__(self, label_names):
        self.label_names = label_names
    
    def _flatten(self, list_of_lists):
        flattened = [val for sublist in list_of_lists for val in sublist]
        return flattened

    def __call__(self, logits_and_labels):
        logits, labels = logits_and_labels
        preds=np.argmax(logits, axis=-1)

        str_labels=[
            [self.label_names[t] for t in label if t!=-100] for label in labels
        ]

        str_preds = [
            [self.label_names[p] for p, t in zip(pred, targ) if t!=-100] for pred, targ in zip(preds, labels)
        ]
        labels_flat = self._flatten(str_labels)
        preds_flat = self._flatten(str_preds)

        acc = accuracy_score(labels_flat, preds_flat)
        f1 = f1_score(labels_flat, preds_flat, average='macro')

        return {
            'accuracy': acc,
            'f1_score': f1
        }
