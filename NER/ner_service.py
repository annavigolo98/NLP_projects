from pydantic import BaseModel
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from NER.data_processor import DataProcessor
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer

from NER.metric_evaluator import MetricEvaluator

class NERService(BaseModel):

    def handle_NER(self, n_epochs):
        '''logic to implement Named Entity Recognition'''
        dataset = load_dataset('conll2003', trust_remote_code=True)

        id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 
                        5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
        
        label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 
                        'B-MISC': 7, 'I-MISC': 8}
 
        checkpoint = 'distilbert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        data_processor = DataProcessor()
        tokenized_dataset = data_processor.tokenizer(dataset, tokenizer)
        
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
        model = AutoModelForTokenClassification.from_pretrained(checkpoint, 
                                                                id2label=id2label,
                                                                label2id=label2id)
        
        metric_evaluator = MetricEvaluator()

        training_arguments = TrainingArguments(
            'NER/checkpoints',
            eval_strategy = 'epoch', 
            save_strategy = 'epoch',
            learning_rate = 2e-05,
            num_train_epochs = n_epochs,
            weight_decay = 0.02
        )

        trainer = Trainer(
            model = model,
            args = training_arguments,
            train_dataset = tokenized_dataset['train'],
            eval_dataset = tokenized_dataset['validation'],
            data_collator = data_collator,
            compute_metrics = metric_evaluator.compute_metrics,
            tokenizer = tokenizer
            )
        
        trainer.train()
        trainer.save_model('NER/saved_model')






        

  
        
        
                    
