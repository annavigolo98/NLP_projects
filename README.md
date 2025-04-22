# NLP
Some exercises from the course: "Data Science: Transformers For Natural Language Processing", by lazy programmer on UDEMY.

The purpose is revisiting their notebooks and lessons to better learn the contents of the course and for rewriting the code using OOP. 

These exercises are based on the course lectures and revisited. Some exercises are made from the course's suggestions. The transformer library Hugging Face is used for the processing and fine-tuning 
of the models. 

1) sentiment_analysis analysis: fine tunes a model for sentiment analysis, using the 'glue-sst2' dataset and the 'distilbert-base-uncased' pre-trained checkpoint for sequence classification tasks.
The model is fine tuned to the specific dataset and the model and the epochs checkpoints are saved in the 'sentiments/saved_model' and in the 'sentiments/checkpoints' folders respectively. ('glue', 'sst2') metric is used for evaluating the model performance.

'01_sentiment_analysis_train.py': handles the fine-tuning of the model.
'02_sentiment_analysis_test.py': tests the model performance.


2) NER classifies words in a document: fine tunes a model for NER, using the dataset 'conll2003' and the 'distilbert-base-cased' pre trained checkpoint for token classification tasks. The model is fine tuned and the checkpoints are saved in the 'NER/saved_model' and in the 'NER/checkpoints' folders respectively.
Precision, f1_score, recall and accuracy are used for evaluating the model performance.

'11_NER_train.py': handles the fine-tuning of the model.
'12_NER_test.py': tests the model performance.

3) NER_custom: similar to the NER task but with a different dataset: the 'brown-corpus' dataset, with different names for the labels of the tokens wrt NER. The 'distilbert-base-cased' checkpoint is used for token classification tasks. The model and the checkpoints are saved in the 'NER/saved_model_custom' and in the 'NER/custom_dataset_checkpoints' folders respectively. Accuracy and f1_score are used as evaluation  metrics.

'21_NER_custom_dataset_train.py': handles the fine-tuning of the model.
'22_NER_custom_dataset_test.py': tests the model.

4) Translator: this task involves the fine tuning of a pre trained seq2seq model for language translation. The 'kde4' dataset for translations from English to French is used.
The 'Helsinki-NLP/opus-mt-en-fr' checkpoint is used along with the AutoModelForSeq2SeqLM model from the transformers library to be fine-tuned. The model and the checkpoints are saved in the 'translator/saved_model' and in the 'translator/checkpoints' folders respectively. sacrebleu and bertscore metrics are used for evaluating the model's performance.

'31_translator_train.py': handles the fine-tuning of the model.
'32_translator_test.py': tests the model.


5) Question Answering: a model for extractive question answering task is fine-tuned. The 'squad' dataset is used and the 'distilbert-base-cased' checkpoint is used along with the AutoModelForQuestionAnswering class for fine tuning. The model and the checkpoints are saved in the 'Question_answering/saved_model' and in the 'Question_answering/checkpoints' folders respectively. The 'squad' metric from evaluate library is used to test the model's performance.

'41_QA_train.py': handles the fine-tuning of the model.
'42_QA_test.py': tests the model.

6) Seq2SeqTransformers: model built from scratch using the transformers architecture from the 'Attention is all you need' article. This encoder-decoder model can be trained for the English to Spanish translation task. For the tokenizer, imported from the transformers library, the 'Helsinki-NLP/opus-mt-en-es' checkpoint is used. A custom csv file is used to build the training and test datasets.
CrossEntropy loss and Adam optimizator are used for the training-validation part, and the model is saved in the 'transformer/saved_model/model' folder, while the tokenizer is saved in the 'transformer/saved_model/tokenizer' folder.

'51_transformers_train.py': handles the training of the model.
'52_transformers_test.py': tests the model.

REQUIREMENTS:
'requirements_cpu.txt': used for the requirements in the CPU only machine. 
'requinrements_gpu.txt': used for the requirements in the GPU machine. 
