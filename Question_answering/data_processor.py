from pydantic import BaseModel

class DataProcessor(BaseModel):
    def tokenizer_train(self, dataset, tokenizer, max_length, stride):
        tokenized_train_dataset = dataset.map(lambda batch: self.tokenize_fn_train(batch, 
                                                                            tokenizer, 
                                                                            max_length, 
                                                                            stride), 
                                                        batched=True,
                                                        remove_columns=dataset.column_names)
        return tokenized_train_dataset
    

    def tokenize_fn_train(self, batch, tokenizer, max_length, stride):
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
            offset_mapping = tokenized_batch.pop('offset_mapping')
            original_sample_ids = tokenized_batch.pop('overflow_to_sample_mapping')
            answers = batch['answers']

            start_answer_tokens = []
            end_answer_tokens = []

            for i, offset in enumerate(offset_mapping):
                
                sequence_ids = tokenized_batch.sequence_ids(i)
                start_context_token = sequence_ids.index(1)
                end_context_token = len(sequence_ids) - sequence_ids[::-1].index(1) -1
               
                sample_idx = original_sample_ids[i]
                answer = answers[sample_idx]
                start_answer_char = answer['answer_start'][0]
                end_answer_char = len(answer['text'][0]) + start_answer_char

                start_answer_token, end_answer_token = self._find_answer(offset,
                                                                start_context_token,
                                                                end_context_token,
                                                                start_answer_char,
                                                                end_answer_char)
                start_answer_tokens.append(start_answer_token)
                end_answer_tokens.append(end_answer_token)
            tokenized_batch['start_positions'] = start_answer_tokens
            tokenized_batch['end_positions'] = end_answer_tokens
            return tokenized_batch
    
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
        sample_ids = [] # UUID of each validation sample in the original dataset
        for i in range(len(tokenized_batch['input_ids'])):
            sample_idx = original_samples_idxs[i]
            sample_ids.append(batch['id'][sample_idx])
            sequence_ids = tokenized_batch.sequence_ids(i)
            offset = tokenized_batch['offset_mapping'][i]
            #Mask question with None
            tokenized_batch['offset_mapping'][i] = [x if sequence_ids[j] == 1 else None 
                                                    for j,x in enumerate(offset)]
        
        tokenized_batch['sample_id'] = sample_ids

        return tokenized_batch



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
                i = start_context_window_token
                for start_end_context_char in offset[start_context_window_token:]:
                    start_ctx_char, end_ctx_char = start_end_context_char

                    if start_ctx_char == start_answer_char:
                        start_answer_token = i 

                    if end_ctx_char == end_answer_char:
                        end_answer_token = i 
                        break 
                    i+=1

            return start_answer_token, end_answer_token