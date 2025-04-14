from pydantic import BaseModel 
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
from sentiments.data_processor import DataProcessor
from transformers import AutoModelForSequenceClassification
from transformers import Trainer

from sentiments.metric_evaluator import MetricEvaluator

class SentimentAnalysisService(BaseModel):

    def handle_sentiment_analysis(self):
        dataset = load_dataset('glue', 'sst2')
        checkpoint = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        data_processor = DataProcessor()
        tokenized_dataset = data_processor.tokenizer_function(dataset, tokenizer)
        training_arguments = TrainingArguments(
            'my_trainer',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            num_train_epochs=1
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

        #Save the model
        trainer.save_model('./saved_model')

        
        print(tokenized_dataset)
