from pydantic import BaseModel

class DataProcessor(BaseModel):

    def tokenizer(self, dataset, tokenizer):
        tokenized_dataset = dataset.map(lambda batch: self._tokenize_fn(batch, tokenizer), batched=True, remove_columns=dataset['train'].column_names)
        return tokenized_dataset
    
    def _align_targets(self, word_ids, old_labels):
        idx2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 
                        5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
        
        label2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 
                        'B-MISC': 7, 'I-MISC': 8}
        
        new_labels = []
        last_label = None

        for word in word_ids:
            if word == None:
                label = -100

            elif word != last_label:
                label = old_labels[word]
            
            elif word == last_label:
                label = old_labels[word]
                if idx2label[old_labels[word]].startswith('B'):
                    inner_label = idx2label[old_labels[word]].replace('B', 'I')
                    label = label2idx[inner_label]
            last_label = word 
            new_labels.append(label)
        return new_labels 
    
    def _tokenize_fn(self, batch, tokenizer):
        
        tokenized_batch = tokenizer(batch['tokens'], is_split_into_words=True, truncation=True)
        old_labels_batch = batch['ner_tags'] 
        new_labels_batch = []
        for i, old_labels_item in enumerate(old_labels_batch):
            word_ids = tokenized_batch.word_ids(i)
            new_labels = self._align_targets(word_ids, old_labels_item)
            new_labels_batch.append(new_labels)
        tokenized_batch['labels'] = new_labels_batch 
        return tokenized_batch
    
    