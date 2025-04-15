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

    def handle_NER(self):
        '''logic to implement Named Entity Recognition'''
        dataset = load_dataset('conll2003', trust_remote_code=True)
        #print(dataset)
        print('Positional tags: ',dataset['train'][0]['pos_tags'])
        print('Chunk tags: ',dataset['train'][0]['chunk_tags'])
        print('NER tags: ',dataset['train'][0]['ner_tags'])


        id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 
                        5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
        
        label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 
                        'B-MISC': 7, 'I-MISC': 8}

        #print('Unique NER tags: ', dataset['train'].features['ner_tags'].feature.names)
        #['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        # Load the tokenizer 
        checkpoint = 'distilbert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        #tokenized_example = tokenizer(dataset['train'][5]['tokens'], is_split_into_words=True)
        #print('Word ids (i.e. indexes of the words in the sentence before tokenization: )', tokenized_example.word_ids())
        data_processor = DataProcessor()
        tokenized_dataset = data_processor.tokenizer(dataset, tokenizer)
        print(tokenized_dataset)


        #EXAMPLE 
        #old_labels_example = [7, 0, 0, 0, 3]
        #words_example = ['[CLS]', 'Ger', '##man', 'call', 'to', 'boycott', 'Micro', '##soft', '[SEP]']
        #word_ids_example = [None, 0, 0, 1, 2, 3, 4, 4, None]
        #idx2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 
        #                 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
            
        #new_labels = data_processor._align_targets(word_ids_example, old_labels_example)

        #new_label_names = [idx2label[new_label] if new_label != -100 else -100 for new_label in new_labels]

        #for word, label_name in zip(words_example, new_label_names):
        #    print(word, label_name) 
        
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
        model = AutoModelForTokenClassification.from_pretrained(checkpoint, 
                                                                id2label=id2label,
                                                                label2id=label2id)
        
        metric_evaluator = MetricEvaluator()

        training_arguments = TrainingArguments(
            'distilbert-finetuned-ner',
            evaluation_strategy = 'epoch', 
            save_strategy = 'epoch',
            learning_rate = 2e-05,
            num_train_epochs = 3,
            weight_decay = 0.01
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
        #save the model
        trainer.save_model('NER/saved_model')
        print('Something')






        

  
        
        
                    
