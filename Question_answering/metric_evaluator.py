import evaluate 
from tqdm import tqdm 

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
            
            n_largest = 20 # max number of logits we want to look at

            max_answer_length = 30 #maximum number of token we want to find in the answer
            predicted_answers = []

            for sample in tqdm(orig_dataset):
                sample_id = sample['id']
                context = sample['context']

                best_score = float('-inf')
                best_answer = None 

                for idx in sample_id2idxs[sample_id]:
                    start_logit = start_logits[idx]
                    end_logit = end_logits[idx]
                    offsets = processed_dataset['offset_mapping'][idx]
                    start_indices = (-start_logit).argsort()
                    end_indices = (-end_logit).argsort()

                    for start_idx in start_indices[:n_largest]:
                        for end_idx in end_indices[:n_largest]:
                            if offsets[start_idx] is None or offsets[end_idx] is None:
                                continue 
                            if end_idx < start_idx:
                                continue 
                            if end_idx-start_idx+1 > max_answer_length:
                                continue 
                            score = start_logit[start_idx] + end_logit[end_idx]

                            if score > best_score:
                                best_score = score 
                                first_ch = offsets[start_idx][0]
                                last_ch = offsets[end_idx][1]
                                best_answer = context[first_ch:last_ch]

                predicted_answers.append({'id': sample_id, 'prediction_text': best_answer})


            true_answers = [
                {'id': x['id'], 'answers': x['answers']} for x in orig_dataset
                ]
            
            metric = evaluate.load('squad')
            return metric.compute(predictions=predicted_answers, references=true_answers)