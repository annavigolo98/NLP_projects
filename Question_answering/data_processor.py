import numpy as np
from pydantic import BaseModel

class DataProcessor(BaseModel):

    def tokenizer_train(self,
                        dataset, 
                        tokenizer, 
                        max_length, 
                        stride):
        
        tokenized_train_dataset = dataset.map(lambda batch: self.tokenize_fn_train(batch, 
                                                                            tokenizer, 
                                                                            max_length, 
                                                                            stride), 
                                                        batched=True,
                                                        remove_columns=dataset.column_names)
        return tokenized_train_dataset
    

    def tokenize_fn_train(self, batch, tokenizer, max_length, stride):
            questions = [question.strip() for question in batch['question']]

            tokenized_batch = tokenizer(questions,
                                        batch['context'],
                                        max_length=max_length,
                                        truncation='only_second',
                                        stride=stride,
                                        return_overflowing_tokens=True,
                                        return_offsets_mapping=True,
                                        padding='max_length'
                                        )
            
            offset_mapping = tokenized_batch.pop('offset_mapping')
            original_sample_ids = tokenized_batch.pop('overflow_to_sample_mapping')
            answers = batch['answers']

             
            answer_tokens = np.array([self._find_answer(offset,
                                                self._calculate_start_end_context_tokens(tokenized_batch.sequence_ids(i))[0],
                                                self._calculate_start_end_context_tokens(tokenized_batch.sequence_ids(i))[1],
                                                self._calculate_start_end_answer_char(original_sample_ids[i], answers)[0],
                                                self._calculate_start_end_answer_char(original_sample_ids[i], answers)[1])
                                                for i, offset in enumerate(offset_mapping)])

            tokenized_batch['start_positions'] = answer_tokens[:,0]
            tokenized_batch['end_positions'] = answer_tokens[:,1]
            return tokenized_batch
    

    def _calculate_start_end_context_tokens(self, sequence_ids):
         start_context_token = sequence_ids.index(1)
         end_context_token = len(sequence_ids) - sequence_ids[::-1].index(1) -1
         return start_context_token, end_context_token 
    
    def _calculate_start_end_answer_char(self, original_sample_ids, answers):
        sample_idx = original_sample_ids
        answer = answers[sample_idx]
        start_answer_char = answer['answer_start'][0]
        end_answer_char = len(answer['text'][0]) + start_answer_char
        return start_answer_char, end_answer_char
    

    def _find_answer(self,
                    offset, 
                    start_context_window_token,
                    end_context_window_token,
                    start_answer_char,
                    end_answer_char):
            
            start_answer_token = 0
            end_answer_token = 0

            if offset[start_context_window_token][0] > start_answer_char or offset[end_context_window_token][1] < end_answer_char:
                pass
            
            else:
                start_answer_token = next((i for i, start_end_ctx_char in enumerate(offset[start_context_window_token:]) 
                                      if start_end_ctx_char[0] == start_answer_char),None)
                
                end_answer_token = next((i for i, start_end_ctx_char in enumerate(offset[start_context_window_token:]) 
                                      if start_end_ctx_char[1] == end_answer_char),None)
                
            return start_answer_token, end_answer_token
    

    def tokenizer_validation(self, dataset, tokenizer, max_length, stride):
        tokenized_validation_dataset = dataset.map(lambda batch: self.tokenize_fn_validation(batch, 
                                                                            tokenizer, 
                                                                            max_length, 
                                                                            stride), 
                                                        batched=True,
                                                        remove_columns=dataset.column_names)
        return tokenized_validation_dataset


    def tokenize_fn_validation(self, batch, tokenizer, max_length, stride):
        questions = [q.strip() for q in batch['question']]

        tokenized_batch = tokenizer(questions,
                                    batch['context'],
                                    max_length=max_length,
                                    truncation='only_second',
                                    stride=stride,
                                    return_overflowing_tokens=True,
                                    return_offsets_mapping=True,
                                    padding='max_length'
                                    )
        
        original_samples_idxs = tokenized_batch.pop('overflow_to_sample_mapping')
        sample_ids = [batch['id'][original_samples_idxs[i]] 
                      for i in range(len(tokenized_batch['input_ids']))] 
        # UUID of each validation sample in the original dataset

        for i in range(len(tokenized_batch['input_ids'])):
            sequence_ids = tokenized_batch.sequence_ids(i)
            offset = tokenized_batch['offset_mapping'][i]
            #Mask question with None
            tokenized_batch['offset_mapping'][i] = [x if sequence_ids[j] == 1 else None 
                                                    for j,x in enumerate(offset)]
        
        tokenized_batch['sample_id'] = sample_ids

        return tokenized_batch
    

    



    