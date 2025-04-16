from pydantic import BaseModel 
from datasets import load_dataset
from transformers import AutoTokenizer

from translator.data_processor import DataProcessor
from translator.metric_evaluator import MetricEvaluator
from translator.plotter import Plotter
from transformers import AutoModelForSeq2SeqLM 
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import tempfile

class TranslatorService(BaseModel):
    def handle_translations(self, seed):
        '''Class for fine tuning for translations tasks'''
        dataset = load_dataset('kde4', lang1='en', lang2='fr', trust_remote_code=True)
        small_dataset = dataset['train'].shuffle(seed=seed).select(range(1_000)) # Take only the first 1000 elements
        #print(small[0]['translation'])
        splitted_dataset = small_dataset.train_test_split(seed=seed)

        checkpoint = 'Helsinki-NLP/opus-mt-en-fr'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        #Example 
        en = splitted_dataset['train'][5]['translation']['en']
        fr = splitted_dataset['train'][5]['translation']['fr']

        en_tokenized = tokenizer(en)
        fr_tokenized = tokenizer(text_target = fr)

        #print('English tokenized example: ', en_tokenized)
        #print('French tokenized example: ', fr_tokenized)

        # Convert to tokens 
        #print('English tokenized words: ', tokenizer.convert_ids_to_tokens(en_tokenized['input_ids']))
        #print('French tokenized words: ', tokenizer.convert_ids_to_tokens(fr_tokenized['input_ids']))

        plotter = Plotter()
        #plotter.plot_len_sentence_histogram(splitted_dataset, 'en')
        #plotter.plot_len_sentence_histogram(splitted_dataset, 'fr')

        max_input_length = 128  # Plot the histograms of the lengths of the sentences to decide it
        max_target_length = 128 
        
        data_processor = DataProcessor()
        tokenized_dataset = data_processor.tokenizer(splitted_dataset, tokenizer, max_input_length, max_target_length)

        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        evaluator = MetricEvaluator(tokenizer, translated_language='fr')

        tmp_dir = tempfile.mkdtemp()

        training_arguments = Seq2SeqTrainingArguments(
            output_dir=tmp_dir,
            eval_strategy='no', # avoid it here, very time consuming
            save_strategy='epoch',
            learning_rate=2e-05,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=3,
            predict_with_generate=True, # During evaluation, do not use the targets. Let it generate by itself.
            fp16=True # Save some memory
            )
        
        trainer = Seq2SeqTrainer(
            model,
            training_arguments,
            train_dataset = tokenized_dataset['train'],
            eval_dataset = tokenized_dataset['test'],
            data_collator=data_collator,
            tokenizer = tokenizer,
            compute_metrics = evaluator
            )
        
        print(trainer.evaluate(max_length=max_target_length))

        trainer.train()
        
        print(trainer.evaluate(max_length=max_target_length))

        #Save the model
        trainer.save_model('translator/saved_model')

