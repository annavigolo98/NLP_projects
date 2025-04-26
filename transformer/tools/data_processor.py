import numpy as np
from pydantic import BaseModel

class DataProcessor(BaseModel):
    def train_test_split(self,
                        raw_dataset,
                        test_size,
                        seed):
        
        split = raw_dataset['train'].train_test_split(test_size=test_size, seed=seed)
        return split
    
    def tokenizer(self, dataset, tokenizer, max_input_length, max_target_length):

        tokenized_dataset = dataset.map(lambda batch: self.preprocess_function(batch,
                                                                               tokenizer,
                                                                               max_input_length,
                                                                               max_target_length),
                                                        batched=True,
                                                        remove_columns=dataset['train'].column_names
                                                        )
        return tokenized_dataset
    

    def preprocess_function(self,
                            batch,
                            tokenizer,
                            max_input_length,
                            max_target_length):
        
        model_inputs = tokenizer(
            batch['en'], max_length=max_input_length, truncation=True
            )

        labels = tokenizer(
            text_target=batch['es'], max_length=max_target_length, truncation=True
            )

        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    


    def calculate_max_sentence_length(self, dataset, input_choice, target_choice):
        #Calculate the maximum number of tokens to take in the tokenized sentence
        #The cut is performed at the value below which the 90 % of senetence lengths lays.
        
        input_lens = [len(tr[input_choice]) for tr in dataset]
        target_lens = [len(tr[target_choice]) for tr in dataset]
        input_max_length = int(np.percentile(input_lens, 90))
        target_max_length = int(np.percentile(target_lens, 90))
        return input_max_length, target_max_length