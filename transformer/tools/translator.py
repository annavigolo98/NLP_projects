from pydantic import BaseModel
import torch

class Translator(BaseModel):

    def translate(self, 
                  input_sentence, 
                  tokenizer, 
                  encoder, 
                  decoder, 
                  device):
            
        encoder_input = tokenizer(input_sentence, return_tensors='pt').to(device)
        encoder_output = encoder(encoder_input['input_ids'], encoder_input['attention_mask'])

        decoder_input_ids = torch.tensor([[65_001]], device=device)
        decoder_attn_mask = torch.ones_like(decoder_input_ids, device=device)

        for _ in range(32):
            dec_output = decoder(
                encoder_output,
                decoder_input_ids,
                encoder_input['attention_mask'],
                decoder_attn_mask
            )

            prediction_id = torch.argmax(dec_output[:, -1, :], axis=-1)

            decoder_input_ids = torch.hstack((decoder_input_ids, prediction_id.view(1,1)))

            decoder_attn_mask = torch.ones_like(decoder_input_ids)

            if prediction_id == 0:
                break

        translation = tokenizer.decode(decoder_input_ids[0, 1:])
        return translation