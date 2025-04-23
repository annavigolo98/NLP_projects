from pydantic import BaseModel 
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
from sentiments.data_processor import DataProcessor
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from sentiments.metric_evaluator import MetricEvaluator

class SentimentAnalysisService(BaseModel):

    def handle_sentiment_analysis(self, n_epochs):
        dataset = load_dataset('glue', 'sst2')
        checkpoint = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        data_processor = DataProcessor()
        tokenized_dataset = data_processor.tokenizer_function(dataset, tokenizer)
        training_arguments = TrainingArguments(
            'sentiments/checkpoints',
            eval_strategy='epoch',
            save_strategy='epoch',
            num_train_epochs=n_epochs
            )
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=2)
        
        metric_evaluator = MetricEvaluator() 

        
        trainer = Trainer(
            model,
            training_arguments,
            train_dataset=tokenized_dataset['train'],
            eval_dataset = tokenized_dataset['validation'],
            tokenizer = tokenizer,
            compute_metrics = metric_evaluator.evaluate_metric
        )
        trainer.train()

        trainer.save_model('sentiments/saved_model')

