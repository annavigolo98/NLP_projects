from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

from transformer.repository.load_data import LoadData
from transformer.tools.data_processor import DataProcessor
from transformer.tools.decoder import Decoder
from transformer.tools.device import Device

from transformer.tools.encoder import Encoder
from transformer.tools.train import Train
from transformer.tools.transformer import Transformer
from transformer.tools.translator import Translator
from transformer.tools.plotter import Plotter

class Seq2SeqService(BaseModel):
    
    def handle_seq2seq(self, n_epochs, seed):
        
        device = Device.get_device()
        print('Device used: ', device)
        dataset_loader = LoadData()
        data_processor = DataProcessor()
        
        raw_dataset = dataset_loader.load_dataset()
        split = data_processor.train_test_split(raw_dataset, test_size=0.3, seed=seed)
        
        model_checkpoint = 'Helsinki-NLP/opus-mt-en-es'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        input_choice = 'en'
        target_choice = 'es'
        max_input_length, max_target_length = data_processor.calculate_max_sentence_length(split['train'],
                                                                                           input_choice,
                                                                                           target_choice)

        tokenized_datasets = data_processor.tokenizer(split, tokenizer, max_input_length, max_target_length)

        data_collator = DataCollatorForSeq2Seq(tokenizer)

        train_loader = DataLoader(
            tokenized_datasets['train'],
            shuffle=True,
            batch_size=64,
            collate_fn=data_collator  #padding and pytorch tensor conversion
        )

        valid_loader = DataLoader(
            tokenized_datasets['test'],
            batch_size=64,
            collate_fn=data_collator
        )

        tokenizer.add_special_tokens({'cls_token': '<s>'})

        encoder_config = {
            'vocaboulary_size': tokenizer.vocab_size +1,
            'max_length': 512,
            'd_k': 16,
            'd_m': 64,
            'heads': 4,
            'n_encoder_blocks': 2,
            'dropout_probability': 0.1
        }

        decoder_config = {
            'vocaboulary_size': tokenizer.vocab_size +1,
            'max_length': 512,
            'd_k': 16,
            'd_m': 64,
            'heads': 4,
            'n_decoder_blocks': 2,
            'dropout_probability': 0.1
        }


        encoder = Encoder(**encoder_config)
        decoder = Decoder(**decoder_config)

        transformer = Transformer(encoder, 
                                  decoder, 
                                  {'encoder_config': encoder_config,
                                    'decoder_config': decoder_config}
                                )

        encoder.to(device)
        decoder.to(device)

        loss_function = nn.CrossEntropyLoss(ignore_index=-100)  
        optimizer = torch.optim.Adam(transformer.parameters())

        
        trainer = Train()
        train_losses, eval_losses, bleu_scores_train, bleu_scores_eval = trainer.train(transformer, 
                                                                                        loss_function, 
                                                                                        optimizer, 
                                                                                        train_loader, 
                                                                                        valid_loader, 
                                                                                        n_epochs=n_epochs,
                                                                                        device=device,
                                                                                        tokenizer=tokenizer)

        print('Train losses: ', train_losses, '\n')
        print('Eval losses: ', eval_losses, '\n')

        print('Train metric Bleu: ', bleu_scores_train, '\n')
        print('Eval metric Bleu: ', bleu_scores_eval, '\n')


        plotter = Plotter()
        plotter.plot_loss(train_losses, eval_losses)
        plotter.plot_metrics(bleu_scores_train, bleu_scores_eval, 'Bleu-score')
        
        transformer.save_pretrained(r'transformer\saved_model\model')
        tokenizer.save_pretrained(r'transformer\saved_model\tokenizer')



    def translate_sentence(self, sentence_to_translate):
        translator = Translator()
        tokenizer = AutoTokenizer.from_pretrained(r'transformer\saved_model\tokenizer')

        _, encoder, decoder = Transformer.from_pretrained(r'transformer\saved_model\model')
        device = Device.get_device()
        encoder.to(device)
        decoder.to(device)
        translation = translator.translate(sentence_to_translate,
                                           tokenizer,
                                           encoder,
                                           decoder,
                                           device
                                           )
        print('Original sentence: ', sentence_to_translate)
        print('Translated sentence: ', translation)
        



        
