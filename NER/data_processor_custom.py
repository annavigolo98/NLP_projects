from pydantic import BaseModel 


class DataProcessorCustom(BaseModel):
    def tokenizer(self, dataset, tokenizer):
        tokenized_datasets = dataset.map(lambda batch: self._tokenize_fn(batch, tokenizer), batched=True, remove_columns=dataset['train'].column_names)
        return tokenized_datasets


    def _align_targets(self, old_labels, word_ids):
        #Mapping label integers to label strings in NER format
        #idx2str = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
        #Mapping  label strings in NER format to label integers
        #str2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
       
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