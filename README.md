# NLP
Some exercises from the course: Data Science: transformers for NLP, by lazy programmer on UDEMY.
These exercises are based on the course lectures and revisited. Some exercises are made from the course's suggestions. The transformer library Hugging Face is used for the processing and fine-tuning 
of the models. 

1) sentiment_analysis analysis: fine tune a model for sentiment analysis, using the 'glue-sst2' dataset and the 'distilbert-base-uncased' pre-trained checkpoint for sequence classification tasks.
The model is finr tuned to the specific dataset and saved the model and the epochs checkpoints in the 'sentiments/saved_model' and in the 'checkpoints' folder respectively. 'glue', 'sst2' metric is used for evaluating the model performance.

'sentiment_analysis_01': handles the fine tuning of the model.
'sentiment_analysis_02': tests the model performance.


2) NER: classifies words in a document: fine tune a model for NER, using the dataset 'conll2003' and the 'distilbert-base-cased' pre trained checkpoint for token classification tasks. The model is fine tuned and the checkpoints are saved in the 'NER/saved_model' and in the 'NER/checkpoints' folders respectively.
Precision, f1_score, recall and accuracy are used for evaluating the model performance.

'NER_01': handles the fine-tuning of the model
'NER_02': tests the model performance 

3) NER_custom: similar to the NER task above but with a different dataset: the 'brown-corpus' dataset, with different names for the labels of the tokens wrt NER. The 'distilbert-base-cased' checkpoint is used for token classification tasks. The model and the checkpoints are saved in the 'NER/saved_model_custom' and in the 'NER/custom_dataset_checkpoints' folders respectively. Accuracy and f1_score are used as evaluation  metrics.

'NER_custom_dataset_01': handles the fine-tuning of the model 
'NER_custom_dataset_02': tests the model 

4) Translator 


5) Question Answering

6) Seq2SeqTransformers