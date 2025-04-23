from pydantic import BaseModel 


class DataProcessorCustom(BaseModel):
    def tokenizer(self, dataset, tokenizer):
        tokenized_datasets = dataset.map(lambda batch: self._tokenize_fn(batch, tokenizer), batched=True, remove_columns=dataset['train'].column_names)
        return tokenized_datasets


    def _align_targets(self, old_labels, word_ids):
        labels = [old_labels[word] if word is not None else -100 for word in word_ids]
        return labels


    def _tokenize_fn(self, batch, tokenizer):
        tokenized_inputs = tokenizer(batch['inputs'], truncation=True, is_split_into_words=True)
        old_targets_batch = batch['targets']
        new_targets_batch = [
            self._align_targets(old_targets, tokenized_inputs.word_ids(i)) 
            for i, old_targets in enumerate(old_targets_batch)
        ]

        tokenized_inputs['labels'] = new_targets_batch
        return tokenized_inputs