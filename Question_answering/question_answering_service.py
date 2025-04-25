from pydantic import BaseModel 
from datasets import load_dataset 
from transformers import AutoTokenizer
from Question_answering.metric_evaluator import MetricEvaluator
from Question_answering.data_processor import DataProcessor 
from transformers import TrainingArguments, Trainer 
from transformers import AutoModelForQuestionAnswering 

class QuestionAnsweringService(BaseModel):
    def handle_question_answering(self, n_epochs):
        '''Handler of QA Task, This extracts the answer to a specific question from a document.'''
        dataset = load_dataset('squad')

        checkpoint='distilbert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        max_length = 380
        stride = 120
        
        data_processor = DataProcessor()
        tokenized_train_dataset = data_processor.tokenizer_train(dataset['train'],
                                                                 tokenizer,
                                                                 max_length,
                                                                 stride)

        tokenized_validation_dataset = data_processor.tokenizer_validation(dataset['validation'],
                                                                 tokenizer,
                                                                 max_length,
                                                                 stride)
        
        arguments = TrainingArguments(
            'Question_answering/checkpoints',
            eval_strategy='no',
            save_strategy='epoch',
            learning_rate=2e-05,
            num_train_epochs=n_epochs,
            weight_decay=0.01,
            fp16=True
        )

        model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)

        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset = tokenized_train_dataset,
            eval_dataset = tokenized_validation_dataset,
            tokenizer=tokenizer
        )

        trainer.train()
        trainer.save_model('Question_answering/saved_model')
        trainer_output = trainer.predict(tokenized_validation_dataset)
        predictions, _, _ = trainer_output 
        start_logits, end_logits = predictions
        
        metric_evaluator = MetricEvaluator()
        computed_metric = metric_evaluator.compute_metrics(start_logits,
                                                           end_logits,
                                                           tokenized_validation_dataset,
                                                           dataset['validation'])
        print('Computed_metric: ', computed_metric, '\n')
        

            



            