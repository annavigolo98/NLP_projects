from pydantic import BaseModel
import torch

class Translator(BaseModel):

    def translate(self, 
                  input_sentence, 
                  tokenizer, 
                  encoder, 
                  decoder, 
                  device):
            
        # get the encoder output first
        print('Device: ', device)
        enc_input = tokenizer(input_sentence, return_tensors='pt').to(device)
        enc_output = encoder(enc_input['input_ids'], enc_input['attention_mask'])

        # setup initial decoder input
        dec_input_ids = torch.tensor([[65_001]], device=device)
        dec_attn_mask = torch.ones_like(dec_input_ids, device=device)

        #decoder loop
        for _ in range(32):
            dec_output = decoder(
                enc_output,
                dec_input_ids,
                enc_input['attention_mask'],
                dec_attn_mask
            )

            #choose the best value
            prediction_id = torch.argmax(dec_output[:, -1, :], axis=-1)

            #append to the decoder input
            dec_input_ids = torch.hstack((dec_input_ids, prediction_id.view(1,1)))

            #recreate the mask
            dec_attn_mask = torch.ones_like(dec_input_ids)

            #exit when reach <\s>
            if prediction_id == 0:
                break

        translation = tokenizer.decode(dec_input_ids[0, 1:])
        return translation