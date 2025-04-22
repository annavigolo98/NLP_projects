from pydantic import BaseModel 
from datasets import load_dataset
from transformers import AutoTokenizer

from translator.data_processor import DataProcessor
from translator.metric_evaluator import MetricEvaluator
from transformers import AutoModelForSeq2SeqLM 
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

class TranslatorService(BaseModel):
    def handle_translations(self, seed):
        '''Class for fine tuning for translations tasks'''
        dataset = load_dataset('kde4', lang1='en', lang2='fr', trust_remote_code=True)
        small_dataset = dataset['train'].shuffle(seed=seed).select(range(1_000)) # Take only the first 1000 elements
        splitted_dataset = small_dataset.train_test_split(seed=seed)

        checkpoint = 'Helsinki-NLP/opus-mt-en-fr'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        max_input_length = 128 
        max_target_length = 128 
        
        data_processor = DataProcessor()
        tokenized_dataset = data_processor.tokenizer(splitted_dataset, tokenizer, max_input_length, max_target_length)

        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        evaluator = MetricEvaluator(tokenizer, translated_language='fr')


        training_arguments = Seq2SeqTrainingArguments(
            output_dir='translator/checkpoints',
            eval_strategy='no',
            save_strategy='epoch',
            learning_rate=2e-05,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=3,
            predict_with_generate=True, 
            fp16=True 
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
        
        evaluate_1 = trainer.evaluate(max_length=max_target_length)

        trainer.train()
        
        evaluate_2 = trainer.evaluate(max_length=max_target_length)
        
        print('Evaluate before training:', evaluate_1)
        print('Evaluate after training:', evaluate_2)
        trainer.save_model('translator/saved_model')

