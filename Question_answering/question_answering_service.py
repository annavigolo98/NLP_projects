from pydantic import BaseModel 
from datasets import load_dataset 
from transformers import AutoTokenizer 


class QuestionAnsweringService(BaseModel):
    def handle_question_answering(self):
        '''Handler of QA Task, This extracts the answer to a specific question from a document.'''
        dataset = load_dataset('squad')
        #print(dataset['train'][0]['answers'])

        #Tokenizer
        checkpoint='distilbert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        #Example of tokenization with question-context

        #context = dataset['train'][0]['context']
        #question = dataset['train'][0]['question']

        #max_length is the maximum length in tokens of a context window
        #stride is the number of overlapping tokens for each context window
        #overflowing_tokens indicate the index of the sample in the original dataset the 
        #subcontext belongs to. 
        #offset_mapping returns the character positions (start, end) of 
        #each word in the question-subcontext pair
        #The characters even in subcontext, refer to the word positions in the original context (non-splitted)


        #tokenized_batch = tokenizer(question,
        #                              context,
        #                              max_length=100,
        #                              truncation='only_second',
        #                              stride=50,
        #                              return_overflowing_tokens=True,
        #                              return_offsets_mapping=True)
        
        #idx=2
        #print('Number of context windows:' , len(tokenized_input), '\n')
        #print('Input ids i_th context window:', tokenized_input['input_ids'][idx], '\n')
        #print('Attention mask i_th context window:', tokenized_input['input_ids'][idx], '\n')
        #print('Offset Mapping i_th context window:', tokenized_input['offset_mapping'][idx], '\n')
        #print('Sequence ids i_th context window:', tokenized_input.sequence_ids(idx), '\n')

        #Find the start and end tokens of the context in the specific subcontext:
        #sequence_ids_i = tokenized_input.sequence_ids(idx)
        #start_context_token = sequence_ids_i.index(1)
        #end_context_token = len(sequence_ids_i) - sequence_ids_i[::-1].index(1) -1
        #print('Start and end context TOKENS in i_th context windos:', 
        #      start_context_token, ', ', end_context_token, '\n')

        #Find answer in the context window (if completely contained in it)
        #answer = dataset['train'][0]['answers']
        #start_answer_char = answer['answer_start'][0]
        #end_answer_char = len(answer['text'][0]) + start_answer_char
        #print('Start and end answer chars in the original context (not splitted)', 
        #      start_answer_char, ', ', end_answer_char, '\n')

        #Find the char position of the first token in the first context window 
        #(char is relative to original non splitted context)
        #Find the char position of the last token in the first context window 
        #(char is relative to original non splitted context)
        #offset = tokenized_input['offset_mapping'][idx]
        #start_context_char = offset[start_context_token][0]
        #end_context_char = offset[end_context_token][1]

        #print('Start and end chars of the  context in the i_th context window '
        #'(relative to the original context)', start_context_char, ', ', end_context_char, '\n')
        
        #OBJECTIVE: Find the answer position TOKEN relative to each context window (not the entire context)

        def find_answer(offset, 
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
        
        #TEST IT
       
        #start_answer_token, end_answer_token = find_answer(offset,
        #                                                   start_context_token,
        #                                                   end_context_token,
        #                                                   start_answer_char,
        #                                                   end_answer_char)
        #answer = tokenizer.decode(tokenized_input['input_ids'][idx][start_answer_token:end_answer_token+1])
        #print('Founded answer tokens: ', start_answer_token, ', ', end_answer_token, '\n')
        #print('Answer: ', answer, '\n')

        max_length = 384
        stride = 128
        
        def tokenize_fn(batch, tokenizer, max_length, stride):
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
                #Find the start and end tokens of the context in the specific subcontext:
                sequence_ids = tokenized_batch.sequence_ids(i)
                start_context_token = sequence_ids.index(1)
                end_context_token = len(sequence_ids) - sequence_ids[::-1].index(1) -1
                #Find the answer start and end chars
                sample_idx = original_sample_ids[i]
                answer = answers[sample_idx]
                start_answer_char = answer['answer_start'][0]
                end_answer_char = len(answer['text'][0]) + start_answer_char

                start_answer_token, end_answer_token = find_answer(offset,
                                                                start_context_token,
                                                                end_context_token,
                                                                start_answer_char,
                                                                end_answer_char)
                start_answer_tokens.append(start_answer_token)
                end_answer_tokens.append(end_answer_token)
            tokenized_batch['start_positions'] = start_answer_tokens
            tokenized_batch['end_positions'] = end_answer_tokens
            return tokenized_batch
        
        tokenized_train_dataset = dataset['train'].map(lambda batch: tokenize_fn(batch, 
                                                                                 tokenizer, 
                                                                                 max_length, 
                                                                                 stride), 
                                                        batched=True,
                                                        remove_columns=dataset['train'].column_names)

        print(len(tokenized_train_dataset))
            



            