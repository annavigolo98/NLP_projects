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
