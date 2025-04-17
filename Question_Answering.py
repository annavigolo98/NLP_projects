from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
from tqdm.autonotebook import tqdm
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline


def _find_answer_token_idx(
    ctx_start,
    ctx_end,
    ans_start_char,
    ans_end_char,
    offset):

    start_idx = 0
    end_idx = 0

    if offset[ctx_start][0] > ans_start_char or offset[ctx_end][1] < ans_end_char:
        pass

    else:
        i = ctx_start
        for start_end_char in offset[ctx_start:]:
            start, end = start_end_char
            if start == ans_start_char:
                start_idx = i
            if end == ans_end_char:
                end_idx = i
                break
            i+=1

    return start_idx, end_idx



#Now we are ready to tokenize-process the data
# i.e. expand question+context pairs into question+smaller context windows
#Google used this for SQuAD dataset


def _tokenize_fn_train(batch, tokenizer):
    max_length=384
    stride=128
    questions = [q.strip() for q in batch['question']]

    inputs = tokenizer(
        questions,
        batch['context'],
        max_length=max_length,
        truncation='only_second',
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length'
        )
    offset_mapping = inputs.pop('offset_mapping')
    orig_samples_ids = inputs.pop('overflow_to_sample_mapping')
    answers = batch['answers']
    start_idxs, end_idxs = [], []

    for i, offset in enumerate(offset_mapping):
        sample_idx = orig_samples_ids[i]
        answer = answers[sample_idx]

        ans_start_char = answer['answer_start'][0]
        ans_end_char = ans_start_char + len(answer['text'][0])

        sequence_ids = inputs.sequence_ids(i)

        #find start and end of context
        ctx_start = sequence_ids.index(1)
        ctx_end = len(sequence_ids) - sequence_ids[::-1].index(1) -1

        start_idx, end_idx = _find_answer_token_idx(ctx_start, ctx_end, ans_start_char, ans_end_char, offset)
        start_idxs.append(start_idx)
        end_idxs.append(end_idx)

    inputs['start_positions'] = start_idxs
    inputs['end_positions'] = end_idxs
    return inputs

#Tokenize the validation set differently
#We won't need the targets since we will just compare with the original answer
#also: replace offset mapping with nones in place of the question

def _tokenize_fn_validation(batch, tokenizer):
    max_length=384
    stride=128
    questions = [q.strip() for q in batch['question']]

    inputs = tokenizer(
        questions,
        batch['context'],
        max_length=max_length,
        truncation='only_second',
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length'
        )
    orig_samples_idxs = inputs.pop('overflow_to_sample_mapping')
    sample_ids = []

    for i in range(len(inputs['input_ids'])):
        sample_idx = orig_samples_idxs[i]
        sample_ids.append(batch['id'][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs['offset_mapping'][i]
        inputs['offset_mapping'][i] = [x if sequence_ids[j] == 1 else None for j, x in enumerate(offset)]
        inputs['sample_id'] = sample_ids

    return inputs

#NB: This function will be not called from the trainer

#NB: This function will be not called from the trainer

def _compute_metrics(start_logits, end_logits, processed_dataset, orig_dataset, metric):
    n_largest = 20 #max number of logit to look at
    max_answer_length = 30
    # Map sample UUID to row indices of processed data
    sample_id2idxs = {}
    for i, id_ in enumerate(processed_dataset['sample_id']):
        if id_ not in sample_id2idxs:
            sample_id2idxs[id_] = [i]
        else:
            sample_id2idxs[id_].append(i)

    predicted_answers = []

    for sample in tqdm(orig_dataset):

        sample_id = sample['id']
        context = sample['context']

        # update these as we loop through candidate answers
        best_score = float('-inf')
        best_answer = None

        #loop through the expanded dataset
        for idx in sample_id2idxs[sample_id]:
            start_logit = start_logits[idx]
            end_logit = end_logits[idx]
            #NB: do not do the reverse (slower) processed_dataset['offset_mapping'][idx]
            offsets = processed_dataset[idx]['offset_mapping']

            start_indices = (-start_logit).argsort()
            end_indices = (-end_logit).argsort()

            for start_idx in start_indices[:n_largest]:
                for end_idx in end_indices[:n_largest]:

                    if offsets[start_idx] is None or offsets[end_idx] is None:
                        continue
                    if end_idx < start_idx:
                        continue
                    if end_idx - start_idx + 1 > max_answer_length:
                        continue

                    score = start_logit[start_idx] + end_logit[end_idx]
                    if score > best_score:
                        best_score = score
                        first_ch = offsets[start_idx][0]
                        last_ch = offsets[end_idx][1]
                        best_answer = context[first_ch:last_ch]

        predicted_answers.append({'id': sample_id, 'prediction_text': best_answer})

  #Compute the metrics
    true_answers = [
        {'id': x['id'], 'answers': x['answers']} for x in orig_dataset
        ]

    return metric.compute(predictions=predicted_answers, references=true_answers)




raw_datasets = load_dataset('squad')
#Ensure that in the training set there is always one answer per sample

small_raw_datasets_train = raw_datasets['train'].shuffle(seed=42).select(range(1000))
small_raw_datasets_train.filter(lambda x: len(x['answers']['text']) != 1)

small_raw_datasets_validation = raw_datasets['validation'].shuffle(seed=42).select(range(1000))
small_raw_datasets_validation.filter(lambda x: len(x['answers']['text']) != 1)

#tokenizer 
model_checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenized_train_dataset = small_raw_datasets_train.map(lambda batch: _tokenize_fn_train(batch, tokenizer), batched=True, remove_columns=small_raw_datasets_train.column_names)
print('Lenght of the raw training dataset and of the tokenized trained dataset respectively: ', 
        len(small_raw_datasets_train), len(tokenized_train_dataset))

tokenized_validation_dataset = small_raw_datasets_validation.map(lambda batch: _tokenize_fn_validation(batch, tokenizer), batched=True, remove_columns=small_raw_datasets_validation.column_names)
print('Lenght of the raw validation dataset and of the tokenized validation dataset respectively: ', 
        len(small_raw_datasets_validation), len(tokenized_validation_dataset))

metric = evaluate.load('squad')
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

args = TrainingArguments(
    'finetined-squad',
    evaluation_strategy='no',
    save_strategy='epoch',
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True
    )

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset.shuffle(seed=42).select(range(1_000)),  #tokenized datasets
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer
    )

#trainer.train()
#trainer_output = trainer.predict(tokenized_validation_dataset)
#Extract the logits 
#predictions, _, _ = trainer_output
#start_logits, end_logits = predictions

#_compute_metrics(
#    start_logits,
#    end_logits,
#    validation_dataset,
#    small_validation_datasets_validation,
#    metric
#    )

#trainer.save_model('my_saved_model_QA')

#qa = pipeline('question-answering',
#          model='my_saved_model_QA',
#          device=0)
#context = 'Today I went to the store to purchase a carton of milk'
#print(qa(context=context, question='What did I buy?'))

    

