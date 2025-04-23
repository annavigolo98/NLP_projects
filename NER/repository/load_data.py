from pydantic import BaseModel
from datasets import load_dataset
import nltk
from nltk.corpus import brown
import json

class LoadData(BaseModel):
    def load_custom_dataset(self, seed, label2idx):
        nltk.download('brown')
        nltk.download('universal_tagset')

        corpus = brown.tagged_sents(tagset='universal')

        json_dict = [{'inputs': [str(word) for (word, _) in sentence],
                            'targets': [label2idx[label] for (_, label) in sentence]} for sentence in corpus]

        with open('NER/data/NER_custom_data.json', 'w') as out_file:
            json.dump(json_dict, out_file)

        data = load_dataset('json', data_files='NER/data/NER_custom_data.json')
        splitted_dataset = data['train'].train_test_split(test_size=0.3, seed=seed)
        return splitted_dataset
