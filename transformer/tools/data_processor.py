from pydantic import BaseModel

class DataProcessor(BaseModel):
    def train_test_split(self, raw_dataset, test_size):
        split = raw_dataset['train'].train_test_split(test_size=test_size, seed=42)
        return split
    
    def tokenizer(self, dataset, tokenizer):
        tokenized_dataset = dataset.map(lambda batch: self.preprocess_function(batch, tokenizer),
                                                        batched=True,
                                                        remove_columns=dataset['train'].column_names
                                                        )
        return tokenized_dataset
    

    def preprocess_function(self,
                            batch,
                            tokenizer):
        max_input_length = 120
        max_target_length = 120
        model_inputs = tokenizer(
            batch['en'], max_length=max_input_length, truncation=True
            )

        labels = tokenizer(
            text_target=batch['es'], max_length=max_target_length, truncation=True
            )

        model_inputs['labels'] = labels['input_ids']
        return model_inputs