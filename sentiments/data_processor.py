from pydantic import BaseModel

class DataProcessor(BaseModel):
    def tokenizer_function(self, dataset, tokenizer):

        def tokenize_fn(batch):
            tokenized_dataset = tokenizer(batch['sentence'], truncation=True)
            return tokenized_dataset
        
        tokenized_dataset = dataset.map(tokenize_fn, batched=True)
        return tokenized_dataset
    