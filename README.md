# NLP
This repository contains a set of NLP projects built using the Hugging Face `transformers` library.  
These projects are inspired by exercises and concepts from the Udemy course  
**"Data Science: Transformers For Natural Language Processing" by Lazy Programmer Inc.**.
This is a personal learning repo.

The purpose of this repo is to:
- Revisit and apply lessons from the course
- Refactor and extend code using object-oriented programming (OOP) principles
- Train and evaluate transformer models on various NLP tasks



These exercises are based on the course lectures and revisited. Some exercises are made from the course's suggestions. The transformer library Hugging Face is used for the processing and fine-tuning 
of the models. 

1) sentiment analysis fine tuning: 

-DATASET: 'glue-sst2'.

-MODEL: 'distilbert-base-uncased' pre-trained checkpoint for sequence classification tasks.

-SAVED: 'sentiments/saved_model', 'sentiments/checkpoints'.

-METRIC: ('glue', 'sst2') metric.

-FILES: '01_sentiment_analysis_train.py': handles the fine-tuning of the model,       '02_sentiment_analysis_test.py': tests the model performance.



2) NER fine tuning: 

-DATASET: 'conll2003'.

-MODEL: 'distilbert-base-cased' pre trained checkpoint for token classification tasks.

-SAVED: 'NER/saved_model', 'NER/checkpoints'.

-METRIC: precision, f1_score, recall, accuracy.

-FILES: '11_NER_train.py': handles the fine-tuning of the model, 
'12_NER_test.py': tests the model performance.

3) NER_custom fine tuning: 

-DATASET: 'brown-corpus'. 

-MODEL: 'distilbert-base-cased' checkpoint for token classification tasks.

-SAVED:  'NER/saved_model_custom', 'NER/custom_dataset_checkpoints'.

-METRIC: accuracy, f1_score. 

-FILES: '21_NER_custom_dataset_train.py': handles the fine-tuning of the model,
'22_NER_custom_dataset_test.py': tests the model.

4) Translator fine tuning: 

-DATASET: 'kde4'.

-MODEL: 'Helsinki-NLP/opus-mt-en-fr' checkpoint for seq2seq tasks.

-SAVED: 'translator/saved_model', 'translator/checkpoints'.

-METRIC: sacrebleu, bertscore.

-FILES: '31_translator_train.py': handles the fine-tuning of the model,
'32_translator_test.py': tests the model.


5) Question Answering fine tuning: 

-DATASET: 'squad'.

-MODEL: 'distilbert-base-cased' checkpoint for extractive question-answering.

-SAVED: 'Question_answering/saved_model', 'Question_answering/checkpoints'.

-METRIC: 'squad' metric.

-FILES: '41_QA_train.py': handles the fine-tuning of the model,
'42_QA_test.py': tests the model.

6) Seq2SeqTransformers: model built from scratch using the transformers architecture. This encoder-decoder model can be trained for the English to Spanish translation task. 

-DATASET: a custom csv file is used to build the training and test datasets.

-MODEL: encoder-decoder transformer architechture from the 'Attention is all you need' article.

-SAVED: 'transformer/saved_model/model', 'transformer/saved_model/tokenizer'.

-FILES: '51_transformers_train.py': handles the training of the model,
'52_transformers_test.py': tests the model.


REQUIREMENTS:

'requirements_cpu.txt': used for the requirements in the CPU only machine. 

'requirements_gpu.txt': used for the requirements in the GPU machine. 
