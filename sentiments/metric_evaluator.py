from pydantic import BaseModel 
import evaluate 
import numpy as np

class MetricEvaluator(BaseModel):
    def evaluate_metric(self, logits_and_labels):
        metric = evaluate.load('glue','sst2')
        logits, labels = logits_and_labels
        predictions = np.argmax(logits, axis=1)
        return metric.compute(predictions=predictions, references=labels)
    