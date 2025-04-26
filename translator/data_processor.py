import numpy as np
from pydantic import BaseModel 

class DataProcessor(BaseModel):

    def tokenizer(self,
                dataset,
                tokenizer, 
                max_input_length, 
                max_target_length):
        
        tokenized_dataset = dataset.map(lambda batch: self._tokenize_fn(batch, 
                                                                        tokenizer, 
                                                                        max_input_length, 
                                                                        max_target_length), 
                                                    batched=True,
                                                    remove_columns=dataset['train'].column_names)
        return tokenized_dataset
        
    def _tokenize_fn(self, batch, tokenizer, max_input_length, max_target_length):
        
        inputs = [x['en'] for x in batch['translation']]
        targets = [x['fr'] for x in batch['translation']]

        tokenized_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        tokenized_targets = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

        tokenized_inputs['labels'] = tokenized_targets['input_ids']
        return tokenized_inputs
    

    def calculate_max_sentence_length(self, dataset, input_choice, target_choice):
        #Calculate the maximum number of tokens to take in the tokenized sentence
        #The cut is performed at the value below which the 90 % of senetence lengths lays.
        
        train = dataset['translation']
        input_lens = [len(tr[input_choice]) for tr in train]
        target_lens = [len(tr[target_choice]) for tr in train]
        input_max_length = int(np.percentile(input_lens, 90))
        target_max_length = int(np.percentile(target_lens, 90))
        return input_max_length, target_max_length
