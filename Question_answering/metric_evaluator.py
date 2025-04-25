import evaluate 
import numpy as np
from tqdm import tqdm 
from itertools import product

class MetricEvaluator:

    def compute_metrics(
                self,
                start_logits, 
                end_logits,
                processed_dataset,
                orig_dataset
                ):
            

            sample_id2idxs = {}
    
            for i, id_ in enumerate(processed_dataset['sample_id']):
                if id_ not in sample_id2idxs:
                    sample_id2idxs[id_] = [i]
                else:
                    sample_id2idxs[id_].append(i)
            
            n_largest = 30 # max number of logits we want to look at
            max_answer_length = 30 #maximum number of token we want to find in the answer
            predicted_answers = []

            for sample in tqdm(orig_dataset):
                sample_id = sample['id']
                context = sample['context']

                #best_score = float('-inf')
                best_answer = None 

                for idx in sample_id2idxs[sample_id]:
                    start_logit = start_logits[idx]
                    end_logit = end_logits[idx]
                    offsets = processed_dataset['offset_mapping'][idx]
                    start_indices = np.argsort(start_logit)[:n_largest]
                    end_indices = np.argsort(end_logit)[:n_largest]

                    _, start_idx, end_idx = next(( (start_logit[start_idx] + end_logit[end_idx], start_idx, end_idx )
                                 for start_idx, end_idx in product(reversed(start_indices), reversed(end_indices))
                                 if offsets[start_idx] is not None and offsets[end_idx] is not None
                                 and start_idx <= end_idx and end_idx-start_idx+1 <= max_answer_length), (None, None, None))
                    
                    if start_idx is not None and end_idx is not None:
                        first_ch, last_ch = offsets[start_idx][0], offsets[end_idx][1]
                        best_answer = context[first_ch:last_ch]

                    #for start_idx in reversed(start_indices):
                    #    for end_idx in reversed(end_indices):
                    #        if offsets[start_idx] is None or offsets[end_idx] is None or end_idx < start_idx or end_idx-start_idx+1 > max_answer_length:
                    #            continue 
                            
                    #        score = start_logit[start_idx] + end_logit[end_idx]

                    #        if score > best_score:
                    #            best_score = score 
                    #            first_ch, last_ch = offsets[start_idx][0], offsets[end_idx][1]
                    #            best_answer = context[first_ch:last_ch]

                predicted_answers.append({'id': sample_id, 'prediction_text': best_answer})


            true_answers = [
                {'id': x['id'], 'answers': x['answers']} for x in orig_dataset
                ]
            
            metric = evaluate.load('squad')
            return metric.compute(predictions=predicted_answers, references=true_answers)