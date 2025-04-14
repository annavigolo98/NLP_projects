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

class Seq2SeqService(BaseModel):
    
    def handle_seq2seq(self):
        
        device = Device.get_device()
        print('Device: ', device)
        dataset_loader = LoadData()
        data_processor = DataProcessor()
        
        raw_dataset = dataset_loader.load_dataset()
        split = data_processor.train_test_split(raw_dataset, test_size=0.3)
        
        #Tokenize datasets 
        model_checkpoint = 'Helsinki-NLP/opus-mt-en-es'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        tokenized_datasets = data_processor.tokenizer(split, tokenizer)
        print(tokenized_datasets)

        data_collator = DataCollatorForSeq2Seq(tokenizer)

        train_loader = DataLoader(
            tokenized_datasets['train'],
            shuffle=True,
            batch_size=64,
            collate_fn=data_collator
        )

        valid_loader = DataLoader(
            tokenized_datasets['test'],
            batch_size=64,
            collate_fn=data_collator
        )

        print(tokenizer.vocab_size)

        #Add start of sentence token
        tokenizer.add_special_tokens({'cls_token': '<s>'})

        print(tokenizer('<s>'))

        encoder_config = {
            'vocab_size': tokenizer.vocab_size +1,
            'max_len': 512,
            'd_k': 16,
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'dropout_prob': 0.1
        }

        decoder_config = {
            'vocab_size': tokenizer.vocab_size +1,
            'max_len': 512,
            'd_k': 16,
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'dropout_prob': 0.1
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

        #Loss and Optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=-100)  # ignore pad tokens in the labels (here represented by -100)
        optimizer = torch.optim.Adam(transformer.parameters())

        
        trainer = Train()
        train_losses, test_losses = trainer.train(transformer, 
                                                  criterion, 
                                                  optimizer, 
                                                  train_loader, 
                                                  valid_loader, 
                                                  epochs=15,
                                                  device=device,
                                                  tokenizer=tokenizer)
        

        print('Train losses: ', train_losses, '\n')
        print('Test losses: ', test_losses)
        #Save Model
        transformer.save_pretrained(r'transformer\saved_model\model')
        tokenizer.save_pretrained(r'transformer\saved_model\tokenizer')



    def translate_sentence(self, sentence_to_translate):
        #Translate a sentence to test the model
        translator = Translator()
        #load tokenizer 
        tokenizer = AutoTokenizer.from_pretrained(r'transformer\saved_model\tokenizer')

        #odel_checkpoint = 'Helsinki-NLP/opus-mt-en-es'
        #okenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        #Load model
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
        



        
