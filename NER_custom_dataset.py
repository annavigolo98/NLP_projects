import numpy as np
import pprint
import json

#DATASETS
from datasets import load_dataset
import nltk
from nltk.corpus import brown

#TOKENIZER
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline

#!pip install transformers datasets
#!pip install opencv-python

#CODE STARTS HERE 

nltk.download('brown')
nltk.download('universal_tagset')

corpus = brown.tagged_sents(tagset='universal')
print(len(corpus))

set_labels = set([item[1] for sublist in corpus for item in sublist])
print(set_labels)
print(corpus[:2])

label2idx = {'NUM': 0, 'CONJ': 1, '.': 2, 'VERB': 3, 'ADV': 4, 'NOUN': 5, 'ADP': 6, 'PRT': 7, 'X': 8, 'PRON': 9, 'DET': 10, 'ADJ': 11}
idx2label = {0: 'NUM', 1: 'CONJ', 2: '.', 3: 'VERB', 4: 'ADV', 5: 'NOUN', 6: 'ADP', 7:'PRT', 8: 'X', 9: 'PRON', 10: 'DET', 11: 'ADJ'}
label_names = [idx2label[key] for key, value in idx2label.items()]
print(label_names)

json_dict = []

for sentence in corpus:
    json_dict.append({'inputs': [str(word) for (word, label) in sentence],
                    'targets': [label2idx[label] for (word, label) in sentence]})


print(json_dict[0],'\n', json_dict[1])

with open('NER_data.json', 'w') as out_file:
    json.dump(json_dict, out_file)


data = load_dataset('json', data_files='NER_data.json')
print(data)
print(data['train'][0])
splitted_dataset = data['train'].train_test_split(test_size=0.3, seed=42)

print(splitted_dataset['test'][0])


checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
trial = tokenizer(splitted_dataset['train'][0]['inputs'], is_split_into_words=True)
pprint.pprint(trial)

print(trial.tokens(), '\n')
print(trial.word_ids())

def align_targets(old_labels, word_ids):
    #Mapping label integers to label strings in NER format
    #idx2str = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    #Mapping  label strings in NER format to label integers
    #str2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    #print('ENTERED')
    labels=[]

    previous_word = None
    for word in word_ids:
        if word == None:
            label = -100

        elif word != previous_word:
            label = old_labels[word]

        else:
            label = old_labels[word]

        labels.append(label)
        previous_word = word

        return labels


def tokenize_fn(batch):
    tokenized_inputs = tokenizer(batch['inputs'], truncation=True, is_split_into_words=True)
    #print('TOKENIZED BATCH', tokenized_inputs)
    old_targets_batch = batch['targets']
    #print(len(old_targets_batch))
    new_targets_batch = []
    for i, old_targets in enumerate(old_targets_batch):
        word_ids = tokenized_inputs.word_ids(i)
        new_targets_batch.append(align_targets(old_targets, word_ids))
    #print(len(new_targets_batch))
    tokenized_inputs['labels'] = new_targets_batch
    return tokenized_inputs

def flatten(list_of_lists):
    flattened = [val for sublist in list_of_lists for val in sublist]
    return flattened

def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    preds=np.argmax(logits, axis=-1)

    #remove -100 from labels and predictions
    #and convert labels_ids to label names
    str_labels=[
        [label_names[t] for t in label if t!=-100] for label in labels
    ]
    # Remove -100 only if the true label is -100
    str_preds = [
        [label_names[p] for p, t in zip(pred, targ) if t!=-100] for pred, targ in zip(preds, labels)
    ]
    labels_flat = flatten(str_labels)
    preds_flat = flatten(str_preds)

    acc = accuracy_score(labels_flat, preds_flat)
    f1 = f1_score(labels_flat, preds_flat, average='macro')

    return {
        'accuracy': acc,
        'f1_score': f1
    }

#TRY A CUSTOM SENTENCE TO SEE IF IT WORKS
old_labels = [5, 3, 5]
words = [
'[CLS]', 'Ger', '##man', 'calling', 'Ita', '#ly', '[SEP]']
word_ids = [None, 0, 0, 1, 2, 2, None]

aligned_labels = align_targets(old_labels, word_ids)

for x, y in zip(words, aligned_labels):
    print(f"{x}\t{y}")

tokenized_datasets = splitted_dataset.map(tokenize_fn, batched=True, remove_columns=splitted_dataset['train'].column_names)
tokenized_datasets['train'][0]

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
batch = data_collator([tokenized_datasets['train'][0]])
batch


model = AutoModelForTokenClassification.from_pretrained(
    checkpoint,
    id2label=idx2label,
    label2id=label2idx,
)

training_args = TrainingArguments(
    'distilbert-finetuned-ner',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)


#trainer.train()

#trainer.save_model('my_saved_NER_custom_model')

#pipeline(
#    'token-classification',
#    model='my_saved_NER_custom_model',
#    aggregation_strategy='simple',
#    device=0
#)


#s='I have been living in Germany for 6 months.'
#ner(s)








