from pydantic import BaseModel

from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer

from NER.data_processor_custom import DataProcessorCustom
from NER.metric_evaluator_custom import MetricEvaluatorCustom
from NER.repository.load_data import LoadData

class NERCustomService(BaseModel):
    def handle_custom_ner_dataset(self, seed):

        label2idx = {'NUM': 0, 'CONJ': 1, '.': 2, 'VERB': 3, 'ADV': 4, 'NOUN': 5, 'ADP': 6, 'PRT': 7, 'X': 8, 'PRON': 9, 'DET': 10, 'ADJ': 11}
        idx2label = {0: 'NUM', 1: 'CONJ', 2: '.', 3: 'VERB', 4: 'ADV', 5: 'NOUN', 6: 'ADP', 7:'PRT', 8: 'X', 9: 'PRON', 10: 'DET', 11: 'ADJ'}
        label_names = [idx2label[key] for key, value in idx2label.items()]
        
        load_data = LoadData()
        splitted_dataset = load_data.load_custom_dataset(seed, label2idx)

        checkpoint = 'distilbert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
       
        data_processor = DataProcessorCustom()
        tokenized_datasets = data_processor.tokenizer(splitted_dataset, tokenizer)

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        metric_eval = MetricEvaluatorCustom(label_names)
        
        model = AutoModelForTokenClassification.from_pretrained(
            checkpoint,
            id2label=idx2label,
            label2id=label2idx,
        )

        training_args = TrainingArguments(
            'NER/custom_dataset_checkpoints',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=2e-5,
            num_train_epochs=1,
            weight_decay=0.01
        )


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'].select(range(100)),
            eval_dataset=tokenized_datasets['test'],
            data_collator=data_collator,
            compute_metrics=metric_eval,
            tokenizer=tokenizer
        )


        trainer.train()

        #trainer.save_model('NER/saved_model_custom')

        

