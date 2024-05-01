# eval.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import evaluate
import logging
logger = logging.getLogger(__name__)
class MetricEvaluator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.accuracy_metric = evaluate.load('./evaluate_metrics/accuracy.py')
        self.precision_metric = evaluate.load('./evaluate_metrics/precision.py')
        self.recall_metric = evaluate.load('./evaluate_metrics/recall.py')
        self.cer_metric = evaluate.load('./evaluate_metrics/cer.py')
        self.wer_metric = evaluate.load('./evaluate_metrics/wer.py')
        self.n_detected_words = 0
        self.n_gt_words = 0        
        self.n_match_words = 0

    def get(self,evalutor_name):
        if evalutor_name == 'CER':
            return self.cer_metric
        elif evalutor_name == 'WER':
            return self.wer_metric
        else:
            print('evaluator name is wrong')
            return None
        
    def add_string(self, ref, pred): 
        '''
        ref :str
        pred:str
        '''       
        pred_words = list(pred.split())
        ref_words = list(ref.split())
        self.n_gt_words += len(ref_words)
        self.n_detected_words += len(pred_words)
        for pred_w in pred_words:
            if pred_w in ref_words:
                self.n_match_words += 1
                ref_words.remove(pred_w)
        # self.ref.append(ref)
        # self.pred.append(pred)
        
    def score(self):
        prec = 0 if self.n_detected_words ==0 else  self.n_match_words / float(self.n_detected_words) * 100
        recall = 0 if self.n_gt_words == 0 else self.n_match_words / float(self.n_gt_words) * 100
        f1 = 0 if prec+recall ==0 else 2 * (prec * recall) / (prec + recall)
        return prec, recall, f1

    def reset(self):
        self.n_detected_words = 0
        self.n_gt_words = 0        
        self.n_match_words = 0

    def result_string(self):
        prec, recall, f1 = self.score()
        return f"Precision: {prec:.3f} Recall: {recall:.3f} F1: {f1:.3f}"
    
    def f1_metric(self,pred_str, label_str):
        p = 0
        r = 0
        f1 = 0
        for pred,label in zip(pred_str,label_str):
            self.add_string(label,pred)
            p_, r_, f1_ = self.score()
            p+=p_
            r+=r_
            f1+=f1_
            self.reset()
        p/=len(pred_str)
        r/=len(pred_str)
        f1/=len(pred_str)
        # print(f"Precision: {p:.3f} Recall: {r:.3f} F1: {f1:.3f}")
        return p,r,f1


    def compute_metric(self, pred_ids, label_ids,metric_type='cer'):
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # pred_str:   list of str
        # label_ids : tensor,shape:batch_size,max_len
        try:
            if metric_type == 'acc_p_r_f1':
                average = 'macro'
                acc = self.accuracy_metric.compute(predictions=pred_ids.reshape(1, -1).squeeze(), \
                                                   references=label_ids.reshape(1, -1).squeeze(), \
                                                    normalize=True)['accuracy']
                p = self.precision_metric.compute(predictions=pred_ids.reshape(1, -1).squeeze(), \
                                                  references=label_ids.reshape(1, -1).squeeze(), \
                                                  average=average, zero_division=0)['precision']
                r = self.recall_metric.compute(predictions=pred_ids.reshape(1, -1).squeeze(), \
                                               references=label_ids.reshape(1, -1).squeeze(), \
                                               average=average, zero_division=0)['recall']
                f1 = 0 if p + r == 0 else 2 * p * r / (p + r)
                return {
                    'acc': acc,
                    'precision': p,
                    'recall': r,
                    'f1': f1
                }
            elif metric_type=='cer_acc_p_r_f1':
                cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
                average = 'macro'
                acc = self.accuracy_metric.compute(predictions=pred_ids.reshape(1, -1).squeeze(), \
                                                   references=label_ids.reshape(1, -1).squeeze(), \
                                                   normalize=True)['accuracy']
                p = self.precision_metric.compute(predictions=pred_ids.reshape(1, -1).squeeze(), \
                                                   references=label_ids.reshape(1, -1).squeeze(),\
                                                   average=average,zero_division=0 )['precision']
                r = self.recall_metric.compute(predictions=pred_ids.reshape(1, -1).squeeze(), \
                                                  references=label_ids.reshape(1, -1).squeeze(), \
                                                average=average, zero_division=0)['recall']
                f1 = 0 if p+r ==0 else 2*p*r/(p+r)
                return {
                    'cer':cer,
                    'acc':acc,
                    'precision':p,
                    'recall':r,
                    'f1':f1
                }
            elif metric_type == 'cer':
                cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
                return {
                    'cer':cer
                }
            elif metric_type == 'wer':
                pred_str = [item.lower() for item in pred_str]
                label_str = [item.lower() for item in label_str]
                wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
                return {
                    'wer':wer
                }
            elif metric_type =='cer_acc':
                cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
                acc = self.accuracy_metric.compute(predictions=pred_ids.reshape(1, -1).squeeze(), \
                                              references=label_ids.reshape(1, -1).squeeze(), \
                                                normalize=True)['accuracy']
                return {
                    'cer': cer,
                    'acc':acc
                }
            elif metric_type =='sroie_p_r_f1':
                p = 0
                r = 0
                f1 = 0
                for pred,label in zip(pred_str,label_str):
                    self.add_string(label,pred)
                    p_, r_, f1_ = self.score()
                    p+=p_
                    r+=r_
                    f1+=f1_
                    self.reset()
                p/=len(pred_str)
                r/=len(pred_str)
                f1/=len(pred_str)
                return {
                    'precision': p,
                    'recall': r,
                    'f1': f1,
                }
            elif metric_type =='multi':
                pred_str = [item.lower() for item in pred_str]
                label_str = [item.lower() for item in label_str]
                cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
                wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
                p = 0
                r = 0
                f1 = 0
                for pred,label in zip(pred_str,label_str):
                    self.add_string(label,pred)
                    p_, r_, f1_ = self.score()
                    p+=p_
                    r+=r_
                    f1+=f1_
                    self.reset()
                p/=len(pred_str)
                r/=len(pred_str)
                f1/=len(pred_str)
                return {
                    'precision': p,
                    'recall': r,
                    'f1': f1,
                    'wer':wer,
                    'cer':cer,
                    'acc':1-wer
                }
        except Exception as e:
            logging.info(f"pred_str{pred_str}"
                         f"label_str{label_str}"
                         f"pred_ids{pred_ids}"
                         f"label_ids{label_ids}")
            logging.error(f"An error occurred during evaluation new : {e}")
            precision = -1
            recall = -1
            f1 = -1
            acc = -1
            cer = -1
            wer = -1
            # cer = None
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cer': cer,
                'acc': acc,
                'wer': wer
            }

# usage:
# from eval import MetricEvaluator
# # Assuming `tokenizer` is already initialized and `pred_ids`, `label_ids` are available
# pred_ids must be list of int

# evaluator = MetricEvaluator(tokenizer)
# metrics = evaluator.compute_metric(pred_ids, label_ids)
# print(metrics)

