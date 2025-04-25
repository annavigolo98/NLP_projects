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

                possible_answers = [ self._find_answer(
                                start_logits[idx],
                                end_logits[idx],
                                processed_dataset['offset_mapping'][idx],
                                n_largest,
                                max_answer_length,
                                context
                                ) for idx in sample_id2idxs[sample_id] ]
                
                best_answer = next((answer for answer in possible_answers), None)
                predicted_answers.append({'id': sample_id, 'prediction_text': best_answer})


            true_answers = [
                {'id': x['id'], 'answers': x['answers']} for x in orig_dataset
                ]
            
            metric = evaluate.load('squad')
            return metric.compute(predictions=predicted_answers, references=true_answers)
    

    def _find_answer(self, start_logits, end_logits, offset_mapping, n_largest, max_answer_length, context):
        start_logit = start_logits
        end_logit = end_logits
        offsets = offset_mapping
        start_indices = np.argsort(start_logit)[:n_largest]
        end_indices = np.argsort(end_logit)[:n_largest]

        best_answer = None

        _, start_idx, end_idx = next(( (start_logit[start_idx] + end_logit[end_idx], start_idx, end_idx )
                        for start_idx, end_idx in product(reversed(start_indices), reversed(end_indices))
                        if offsets[start_idx] is not None and offsets[end_idx] is not None
                        and start_idx <= end_idx and end_idx-start_idx+1 <= max_answer_length), (None, None, None))
        
        if start_idx is not None and end_idx is not None:
            first_ch, last_ch = offsets[start_idx][0], offsets[end_idx][1]
            best_answer = context[first_ch:last_ch]

        return best_answer