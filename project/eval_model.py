import torch
import argparse
import os 
import logging
import time 
import pandas as pd 
from tqdm import tqdm
from utils_tools import dict2csv
from eval import MetricEvaluator
from Models import load_model_and_processor
from Dataset import MyDataset
from torch.utils.data import DataLoader
# from run import logger
# Set up basic configuration for logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)





def setup_logging_to_file(filename):
    # 获取根日志记录器
    logger = logging.getLogger()
    # 移除所有已经存在的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # 配置新的处理器
    logging.basicConfig(filename=filename, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logger




# TODO
def Eval_Model(eval_loader,model,device,evaluator):
    model.to(device)
    model.eval()
    pbar = tqdm(eval_loader)
    error_batch = 0
    precision = 0
    recall = 0
    f1_score = 0
    cer = 0
    war = 0
    wer = 0
    num_eval = len(eval_loader)
    with torch.no_grad():
        for batch in pbar :
            try:
                # run batch generation
                pixel_values, labels = batch['pixel_values'].to(device), batch['labels'].to(device)
                outputs = model.generate(pixel_values)
                metric = evaluator.compute_metric(\
                    pred_ids=outputs, label_ids=labels,metric_type='multi')
                precision+=metric['precision']
                recall+=metric['recall']
                f1_score+=metric['f1']
                cer+=metric['cer']
                war+=metric['acc']
                wer+=metric['wer']
                pbar.set_postfix_str(f"f1:{metric['f1']:.3f},cer:{metric['cer']:.3f},war:{metric['acc']:.3f}")
            except Exception as e:
                print(f'during eval a error happened{e}')
                error_batch+=1
    precision/=(num_eval-error_batch)
    recall/=(num_eval-error_batch)
    f1_score/=(num_eval-error_batch)
    cer/=(num_eval-error_batch)
    war/=(num_eval-error_batch)
    wer/=(num_eval-error_batch)
    print(f"error batchs: {error_batch}\n"
        f"Test f1:{f1_score}\n"
        f"Test p:{precision}\n"
        f"Test r:{recall}\n"
        f"Test cer:{cer}\n"
        f"Test war:{war}\n"
        f"Test wer:{wer}\n"
        )
    logging.info(
        f"error batchs: {error_batch}\n"
        f"Test f1:{f1_score}\n"
        f"Test p:{precision}\n"
        f"Test r:{recall}\n"
        f"Test cer:{cer}\n"
        f"Test war:{war}\n"
        f"Test wer:{wer}\n"
        )
    return {
            'precision': precision,
            'recall': recall,
            'f1': f1_score,
            'wer':wer,
            'cer':cer,
            'acc':war
        }

def Eval_model(eval_loader,processor, model,device,evaluator,evaluator_name):
    model.to(device)
    model.eval()
    pbar = tqdm(eval_loader)
    error_batch = 0
    precision = 0
    recall = 0
    f1_score = 0
    num_eval = len(eval_loader)
    if evaluator_name in  ['CER','WER']: 
        metric = evaluator.get(evaluator_name)
    with torch.no_grad():
        for batch in pbar :
            try:
                # run batch generation
                pixel_values, labels = batch['pixel_values'].to(device), batch['labels'].to(device)
                if evaluator_name in  ['CER','WER']: 
                    outputs = model.generate(pixel_values)
                    pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
                    labels[labels == -100] = processor.tokenizer.pad_token_id
                    label_str = processor.batch_decode(labels, skip_special_tokens=True)
                    # compute metrics
                    # print('use CER')
                    # print(pred_str)
                    # print(label_str)
                    if evaluator_name == 'WER':
                        pred_str = [item.lower() for item in pred_str]
                        label_str = [item.lower() for item in label_str]
                    # print(pred_str)
                    # print(label_str)
                    metric.add_batch(predictions=pred_str, references=label_str)
                else:
                    # outputs = model(pixel_values=pixel_values, labels=labels)
                    # pred = torch.argmax(outputs.logits,axis=-1)
                    outputs = model.generate(pixel_values)
                    pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
                    labels[labels == -100] = processor.tokenizer.pad_token_id
                    label_str = processor.batch_decode(labels, skip_special_tokens=True)
                    pred_str = [item.lower() for item in pred_str]
                    label_str = [item.lower() for item in label_str]
                    p,r,f1 = evaluator.f1_metric(pred_str = pred_str,label_str = label_str)
                    precision+=p
                    recall+=r
                    f1_score+=f1
                    pbar.set_postfix_str(f"Test f1:{f1:.4f}")
            except Exception as e:
                print(f'during eval a error happened{e}')
                error_batch+=1
    if evaluator_name in  ['CER','WER']: 
        final_score = metric.compute()
        print(f"error batchs: {error_batch}\n"
          f"Test {evaluator_name}:{final_score}\n"
          f"Test ACC:{1-final_score}\n"
          )
        logging.info(
            f"error batchs: {error_batch}\n"
            f"Test {evaluator_name}:{final_score}\n"
            f"Test ACC:{1-final_score}\n"
          )
    else :
        precision/=(num_eval-error_batch)
        recall/=(num_eval-error_batch)
        f1_score/=(num_eval-error_batch)
        print(f"error batchs: {error_batch}\n"
          f"Test f1:{f1_score}\n"
          f"Test p:{precision}\n"
          f"Test r:{recall}\n"
          )
        logging.info(
            f"error batchs: {error_batch}\n"
            f"Test f1:{f1}\n"
            f"Test p:{precision}\n"
            f"Test r:{recall}\n"
          )
        
        
    
    
    


