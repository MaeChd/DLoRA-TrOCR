import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import torch 
import logging
import time 
import pandas as pd
from utils_tools import list2csv,dict2csv
from TrainEval import train,huggingface_trainer
from eval import MetricEvaluator
from Models import get_model_and_processor,load_model_and_processor
from Dataset import MyDataset
from transformers import AdamW
from transformers import  Seq2SeqTrainingArguments
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from eval_model import Eval_model,Eval_Model
import evaluate
def parse_args():
    '''
    '''
    parser = argparse.ArgumentParser(description="Process some important arguments")
    parser.add_argument('--processor_dir',default='./pretrained-models/trocr-base-stage1/',
                        help='processor directory')
    parser.add_argument('--model_dir', default='./pretrained-models/trocr-base-stage1/',
                        help='pretrained model directory')
    parser.add_argument('--save_dir', default='/root/autodl-tmp/exp/',
                        help='after experiment model save directory')
    parser.add_argument('--data_dir', default='./data/IAM/',
                        help='datasets directory')
    parser.add_argument('--peft_dir', default=None,
                        help='test peft directory')
    # parser.add_argument('--log_file_dir', default='./log_file/',
    #                     help='datasets directory')
    parser.add_argument('--peft_method', default=None,
                        help='method of peft')
    parser.add_argument('--target_modules', default='qkv',
                        help='target_modules of peft')
    parser.add_argument('--rank', default=8,type=int,
                        help='rank of peft')
    parser.add_argument('--alpha', default=32,type=int,
                        help='alpha of peft')
    parser.add_argument('--lr', default=1e-5,type=float,
                        help='learning rate')

    parser.add_argument('--peft_mode', action='store_true', help='Print verbose output.')
    parser.add_argument('--use_trainer', action='store_true', help='Print verbose output.')
    parser.add_argument('--eval_mode', action='store_true', help='Print verbose output.')
    parser.add_argument('--if_config', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--task', default='IAM')
    
    # model config
    parser.add_argument('--max_len', default=32, type=int,help='max_token_length.')

    return parser.parse_args()


def test_pipeline(
        task,
        data_dir,
        processor,
        model,
        device,
        max_len,
        ):
    task_list = TASK_LIST[task]
    precision = 0
    recall = 0
    f1_score = 0
    cer = 0
    war = 0
    wer = 0
    for test_task in task_list:
        print(f'test task:{test_task} start!!')
        logging.info(f'test task:{test_task} start!!')
        
        test_base_dir = os.path.join(data_dir,test_task)
        print(test_base_dir)
        if task == 'IAM':
            test_df = pd.read_fwf(os.path.join(test_base_dir,'gt_test.txt'),header=None)
            test_df.rename(columns={0: "img_name", 1: "text"}, inplace=True)
        else:
            test_df = pd.read_csv(os.path.join(test_base_dir,'gt_test.txt'),sep='\t',header=None)
            test_df.rename(columns={0: "img_name", 1: "text"}, inplace=True)
        test_dataset = MyDataset(root_dir=test_base_dir,df=test_df ,processor=processor,max_target_length=max_len)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        #  evaluator
        evaluator = MetricEvaluator(processor.tokenizer)
        evaluator_name = taskEvalMap[task]
        print(f'evaluator name:{evaluator_name}')
        print("number of evalLoader",len(test_loader))
        print('====val-best model tesing=====')
        logging.info(f"val-best model tesing\n")
        res = Eval_Model(test_loader,model,device,evaluator)
        precision+=res['precision']
        recall+=res['recall']
        f1_score+=res['f1']
        cer+=res['cer']
        war+=res['acc']
        wer+=res['wer']
    precision/=len(task_list)
    recall/=len(task_list)
    f1_score/=len(task_list)
    cer/=len(task_list)
    war/=len(task_list)
    wer/=len(task_list)
    print(
        f"Test f1:{f1_score}\n"
        f"Test p:{precision}\n"
        f"Test r:{recall}\n"
        f"Test cer:{cer}\n"
        f"Test war:{war}\n"
        f"Test wer:{wer}\n"
        )
    logging.info(
        f"Test f1:{f1_score}\n"
        f"Test p:{precision}\n"
        f"Test r:{recall}\n"
        f"Test cer:{cer}\n"
        f"Test war:{war}\n"
        f"Test wer:{wer}\n"
        )


def setup_logging_to_file(filename):
    # 获取根日志记录器
    logger = logging.getLogger()
    # 移除所有已经存在的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # 配置新的处理器
    logging.basicConfig(filename=filename, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logger



target_modules_map = {
        'qkv':['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value'],
        'qv':['q_proj', 'v_proj', 'query', 'value'],
        'qkvo':['q_proj', 'k_proj', 'v_proj','out_proj', 'query', 'key', 'value'],
        'e_qkv':['query', 'key', 'value'],
        'e_qv':['query', 'value'],
        'd_qkv':['q_proj', 'k_proj', 'v_proj'],
        'd_qkvo':['out_proj', 'query', 'key', 'value'],
        'd_qv':['q_proj', 'v_proj'],
        'e_qv_d_qkvo':['query', 'value','q_proj', 'k_proj', 'v_proj','out_proj'],
        'e_qkv_d_qkvo':['query','key' ,'value','q_proj', 'k_proj', 'v_proj','out_proj'],
    }
taskEvalMap  = {
        'IAM':'CER',
        'SROIE':'F1',
        'Latex':'CER',
        'STR':'WER',
        'IAM_SROIE':'Multi',
        'IAM_STR':'Multi',
        'STR_SROIE':'Multi',
        'IAM_STR_SROIE':'Multi',
        'Multi':'Multi',
}
IAM_TEST_LIST = [
            ''
]
SROIE_TEST_LIST = [
    ''
]
LATEX_TEST_LIST = [
    'latex_1147',
    'latex_1199',
    'latex_986'
]
STR_TEST_LIST = [ 
                  'CT80-288',\
                  'IC13/ICDAR2013-1015','IC13/ICDAR2013-1095','IC13/ICDAR2013-857', \
                  'IC15/ICDAR2015-1811','IC15/ICDAR2015-2077',\
                  'III5K',\
                  'SVT',\
                  'SVTP-645'          
               ]

TASK_LIST = {
    'IAM':IAM_TEST_LIST,
    'Latex':LATEX_TEST_LIST,
    'SROIE':SROIE_TEST_LIST,
    'STR':STR_TEST_LIST,
    'IAM_SROIE':IAM_TEST_LIST+SROIE_TEST_LIST,
    'IAM_STR':IAM_TEST_LIST+STR_TEST_LIST,
    'STR_SROIE':STR_TEST_LIST+SROIE_TEST_LIST,
    'IAM_STR_SROIE':IAM_TEST_LIST+STR_TEST_LIST+SROIE_TEST_LIST,
    'Multi':IAM_TEST_LIST+LATEX_TEST_LIST+\
        SROIE_TEST_LIST+STR_TEST_LIST
}
taskDirMap ={
    'IAM':'IAM',
    'Latex':'Latex',
    'SROIE_Task2':'SROIE',
    'STR_BENCHMARKS':'STR'
}
# task_list = ['IAM','SROIE','Latex','STR']
if __name__ =='__main__':
    '''
    
    
    '''
    
    args = parse_args()
    processor_dir = args.processor_dir
    model_dir = args.model_dir ; save_dir = args.save_dir ; data_dir = args.data_dir
    peft_dir =args.peft_dir
    # log_file_dir = args.log_file_dir
    # use lora default False
    peft_mode = args.peft_mode ; peft_method = args.peft_method
    target_modules = args.target_modules ;rank = args.rank
    alpha = args.alpha 
    # use huggingface trainer
    use_trainer = args.use_trainer
    # dataset setting
    task = args.task
    # setting max_token_length
    max_len = args.max_len
    epochs = args.epochs ; lr = args.lr ; batch_size = args.batch_size
    # use eval_mode default False
    eval_mode = args.eval_mode
    only_test = args.only_test
    seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择可用的GPU，如果没有GPU则选择CPU
    # save_dir eg: exp/
    # task eg:IAM
    # peft_method:LORA
    # root_dir = lora_qkv_r_8_a_32_iam_lr_1e-05
    if_config = args.if_config
    if task=='STR':
        if_config = False
    if model_dir.endswith('stage1/'):
        stage = 'stage1'
    else:
        if model_dir.endswith('handwritten/'):
            stage = 'ck_iam'
        elif model_dir.endswith('printed/'):
            stage = 'ck_sroie'
        elif model_dir.endswith('str/'):
            stage = 'ck_str'
        else :
            stage = 'unknow'
    if peft_mode and peft_method is not None:
        root_dir = stage+'_'+peft_method+'_'+target_modules+'_r_'+str(rank)+'_a_'+str(alpha)+'_'+task+'_lr_'+str(lr)
    else :
        root_dir = stage+'_'+'ft_'+task+'_lr_'+str(lr)
    # base_dir eg: exp/lora_qkv_r_8_a_32_iam_lr_1e-05/
    base_dir = os.path.join(save_dir,root_dir)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # build log dir 
    # build save model weight dir 
    log_file_dir = os.path.join(base_dir,'log_file')
    save_val_best_dir = os.path.join(base_dir,'val_best')
    save_min_trainLoss_dir = os.path.join(base_dir,'min_trainLoss')
    # build save folder and logging folder
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)
    if not os.path.exists(save_val_best_dir):
        os.makedirs(save_val_best_dir)
    if not os.path.exists(save_min_trainLoss_dir):
        os.makedirs(save_min_trainLoss_dir)
   
    # settings 
    beam_search_settings= {
        "max_length": max_len,
        # "num_beams": 4,
        # "length_penalty": 2.0,
    }
    model_config_dicts = {
        'beam_search_settings' : beam_search_settings,
    }
    peft_dicts = {
        'peft_method':peft_method,
        'peft_mode':peft_mode,
        'rank':rank,
        # 'target_modules':'.*\.attention\.(attention|self)\.(query|value)$',
        'target_modules':target_modules_map[target_modules],
        'module_alpha':alpha,
        'module_dropout':0.1
    }
    ########################################   training  part #########################################
    print('====loding model and processor====')
    #  target_modules=['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value'],
    processor, model = get_model_and_processor(
                        process_dir=processor_dir,
                        model_dir=model_dir,
                        peft_dict=peft_dicts,
                        model_config_dict=model_config_dicts,
                        if_config = if_config
    )
    if not only_test:
         # Configure global logging settings
        logger = setup_logging_to_file(os.path.join(log_file_dir,\
                                                    'train'+str(epochs)+'_time_'+str(time.time())+'.log'))
        # Example usage of logging in the main script
        logging.info('Training Start')
        print('====lode model and processor successful====')
        print(f'====loding {task} datasets====')
        # load datasets 
        # task:IAM,SROIE,Latex,STR
        # train_val
        # dataset_dir = ../../autodl-tmp/dataset/{data_name}/
        if task=='IAM':
            # IAM have val
            train_df = pd.read_csv(os.path.join(data_dir,'gt_train.txt'),sep='\t',header=None)
            val_df = pd.read_csv(os.path.join(data_dir,'gt_val.txt'),sep='\t',header=None)
            train_df.rename(columns={0: "img_name", 1: "text"}, inplace=True)
            val_df.rename(columns={0: "img_name", 1: "text"}, inplace=True)
            
        else:
            if task =='STR' or task=='SROIE' or task == 'Latex':
                df = pd.read_csv(os.path.join(data_dir,'gt_train.txt'),sep='\t',header=None)
            else:
                task_n = task.lower()
                df = pd.read_csv(os.path.join(data_dir,task_n+'_gt_train.txt'),sep='\t',header=None)
            df.rename(columns={0: "img_name", 1: "text"}, inplace=True)
            train_df, val_df = train_test_split(df, test_size=0.1,random_state=seed)
            # we reset the indices to start from zero
            train_df.reset_index(drop=True, inplace=True)
            val_df.reset_index(drop=True, inplace=True)
            
        train_dataset = MyDataset(root_dir=data_dir,df=train_df,processor=processor,max_target_length=max_len)
        val_dataset = MyDataset(root_dir=data_dir,df=val_df,processor=processor,max_target_length=max_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        print(f'====lode {task} dataset successful====')
        # optimizer ; evaluator
        optimizer = AdamW(model.parameters(), lr=lr)
        evaluator = MetricEvaluator(processor.tokenizer)
        # metrics = evaluator.compute_metric(pred_ids, label_ids)
        print('====start training====')
        print("batch size:",batch_size)
        print("number of trainLoader",len(train_loader))
        print("number of evalLoader",len(val_loader))
        if use_trainer:
            print('use trainer')
            training_args = Seq2SeqTrainingArguments(
                predict_with_generate=True,
                evaluation_strategy="steps",
                save_total_limit = 2,
                load_best_model_at_end = True,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                fp16=True,
                output_dir=save_val_best_dir,
                logging_steps=500,
                save_steps=2000,
                eval_steps=500,
                num_train_epochs=epochs,
                # weight_decay= 0.01,
                report_to='wandb',
                run_name=peft_method,
                metric_for_best_model='eval_score',
                greater_is_better = True
            )
            # cer_metric = evaluate.load('./evaluate_metrics/cer.py')
            def compute_metrics(pred):
                label_ids = pred.label_ids
                pred_ids = pred.predictions
                pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
                label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
                label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
                # cer = cer_metric.compute(predictions=pred_str, references=label_str)
                metric = evaluator.compute_metric(pred_ids=pred_ids,label_ids=label_ids,metric_type='multi')
                
                return {
                    "cer": metric['cer'],
                    "f1": metric['f1'],
                    'score':(100-metric['cer'])+metric['f1']
                }
            trainer = huggingface_trainer(training_args,model,processor,\
                                          train_dataset,val_dataset,compute_metrics,lr)
            trainer.train()
        else :
            train_loss_list, eval_dict  = train(train_loader, 
                                                model,
                                                device,
                                                optimizer,
                                                epochs,
                                                base_dir,# save model dir
                                                task,# must be :IAM,SROIE,Latex,STR,Multi
                                                eval_mode,
                                                val_loader,
                                                evaluator,
                                                peft_mode
                                                )
            # save
            list2csv(train_loss_list,cols=['train loss'],save_dir=os.path.join(log_file_dir+'train_loss.csv'))
            dict2csv(eval_dict,save_dir=os.path.join(log_file_dir+'eval.csv'))
        # end
        print('====end training ====')
    del model
    ########################################   training_part #########################################
