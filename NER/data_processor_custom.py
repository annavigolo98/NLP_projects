from pydantic import BaseModel 


class DataProcessorCustom(BaseModel):
    def tokenizer(self, dataset, tokenizer):
        tokenized_datasets = dataset.map(lambda batch: self._tokenize_fn(batch, tokenizer), batched=True, remove_columns=dataset['train'].column_names)
        return tokenized_datasets


    def _align_targets(self, old_labels, word_ids):
        labels=[]

        previous_word = None
        for word in word_ids:
            if word == None:
                label = -100

            elif word != previous_word:
                label = old_labels[word]

            elif word == previous_word:
                label = old_labels[word]

            labels.append(label)
            previous_word = word

        return labels


    def _tokenize_fn(self, batch, tokenizer):
        tokenized_inputs = tokenizer(batch['inputs'], truncation=True, is_split_into_words=True)
        old_targets_batch = batch['targets']
        new_targets_batch = []
        for i, old_targets in enumerate(old_targets_batch):
            word_ids = tokenized_inputs.word_ids(i)
            new_targets_batch.append(self._align_targets(old_targets, word_ids))
        
        tokenized_inputs['labels'] = new_targets_batch
        return tokenized_inputs