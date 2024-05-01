import torch
import os 
import logging
from transformers import Seq2SeqTrainer
from transformers import default_data_collator,AdamW
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train(train_loader, model, device, optimizer, epochs, save_dir = None,task='IAM',
          eval_mode=False, eval_loader=None, metricEvaluator=None,lora=False
          ):
    if epochs < 1:
        raise ValueError("Epochs must be at least 1.")
    if not hasattr(train_loader, '__iter__'):
        raise ValueError("train_loader must be an iterable.")
    if eval_mode:
        if not hasattr(eval_loader, '__iter__') or metricEvaluator is None:
            raise ValueError("eval_loader and compute_metricor must be provided and iterable when eval_mode is True.")

    train_loss_list = []
    eval_dict = {
        'cer':[],
        'precision':[],
        'recall':[],
        'f1':[],
        'acc':[],
        'wer':[]
    }
    task_mertic_map= {
    'IAM':float('inf'),
    'Latex':float('inf'),
    'SROIE':-float('inf'),
    'STR':float('inf'),
    'Multi':-float('inf'),
    'IAM_SROIE':-float('inf'),
    'IAM_STR':-float('inf'),
    'STR_SROIE':-float('inf'),
    'IAM_STR_SROIE':-float('inf'),
    }

    best_metric = task_mertic_map[task]

    model.to(device)
    min_train_loss = float('inf')
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        train_losses = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as tepoch:
            for batch in tepoch:
                try:
                    pixel_values, labels = batch['pixel_values'].to(device), batch['labels'].to(device)
                    optimizer.zero_grad()  # Clear previous gradients
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    loss.backward()  # Backpropagation
                    optimizer.step()  # Update model parameters
                    train_losses += loss.item()
                    tepoch.set_postfix_str(f"loss:{loss.item():.4f}")
                    # logging.info(f"CER:{metirc.get('cer', -1):.4f} \t"
                    #              f"ACC:{metirc.get('acc', -1):.4f} \t F1:{metirc.get('f1', -1):.4f}")
                except Exception as e:
                    logging.error(f"An error occurred during training: {e}")
                    continue  # Optionally, could break or re-raise
        train_loss = train_losses / len(train_loader)
        train_loss_list.append(train_loss)
        logging.info(f"Epoch: {epoch + 1} \t Loss: {train_loss:.4f} \n ")
        if eval_mode:
            try:
                metric = evals(eval_loader, model, device, metricEvaluator ,task)
                cer = metric.get('cer',0);wer = metric.get('wer',0);precision = metric.get('precision',0)
                recall = metric.get('recall',0);f1 = metric.get('f1',0);acc = metric.get('acc',0)
                eval_dict['cer'].append(cer);eval_dict['wer'].append(wer)
                eval_dict['f1'].append(f1);eval_dict['precision'].append(precision)
                eval_dict['recall'].append(recall);eval_dict['acc'].append(acc)
                # save model 
                if task=='IAM' or task=='Latex':
                    if cer < best_metric:
                        best_metric = cer
                        model.save_pretrained(os.path.join(save_dir ,'val_best'))
                elif task == 'SROIE':
                    if f1>best_metric:
                        best_metric = f1
                        model.save_pretrained(os.path.join(save_dir ,'val_best'))
                elif task =='STR':
                    if wer < best_metric:
                        best_metric = wer
                        model.save_pretrained(os.path.join(save_dir ,'val_best'))
                else :
                    all_metric_value = 1-cer+1-wer+f1+precision+recall
                    if all_metric_value>best_metric:
                        best_metric = all_metric_value
                        model.save_pretrained(os.path.join(save_dir ,'val_best'))
            except Exception as e:
                logging.error(f"An error occurred during using evaluation: {e}")
            if train_loss < min_train_loss :
                min_train_loss = train_loss
                model.save_pretrained(os.path.join(save_dir ,'min_trainLoss'))
    return train_loss_list, eval_dict
  
def evals(eval_loader, model,device,metricEvaluator,task='IAM'):
    model.eval()
    valid_precision = 0.0
    valid_recall = 0.0
    valid_f1 = 0.0
    valid_cer = 0.0
    valid_Accuracy = 0.0
    valid_wer = 0.0
    pbar = tqdm(eval_loader)
    num_eval = len(eval_loader)
    error_batch = 0
    with torch.no_grad():
        if task == 'IAM' or task =='Latex':
            for batch in pbar :
                # run batch generation
                pixel_values, labels = batch['pixel_values'].to(device), batch['labels'].to(device)
                outputs = model.generate(pixel_values)
                # compute metrics
                metirc = metricEvaluator.compute_metric(pred_ids=outputs, label_ids=labels,metric_type='cer')
                if metirc['cer']==-1:
                    print('compute error,skip')
                    error_batch+=1
                else :
                    valid_cer += metirc['cer']
                    pbar.set_postfix_str(f"Test cer:{metirc['cer']:.4f}")
        elif task == 'STR':
            for batch in pbar :
                # run batch generation
                pixel_values, labels = batch['pixel_values'].to(device), batch['labels'].to(device)
                outputs = model.generate(pixel_values)
                # compute metrics

                metirc = metricEvaluator.compute_metric(pred_ids=outputs, label_ids=labels,metric_type='wer')
                
                if metirc['wer']==-1:
                    print('compute error,skip')
                    error_batch+=1
                else :
                    valid_wer += metirc['wer']
                    pbar.set_postfix_str(f"Test wer:{metirc['wer']:.4f}")
        elif task =='SROIE':
            for batch in pbar :
                # run batch generation
                pixel_values, labels = batch['pixel_values'].to(device), batch['labels'].to(device)
                outputs = model.generate(pixel_values)
                # compute metrics
                metirc = metricEvaluator.compute_metric(pred_ids=outputs, label_ids=labels,metric_type='sroie_p_r_f1')
                if metirc['f1']==-1:
                    print('compute error,skip')
                    error_batch+=1
                else :
                    valid_precision += metirc.get('precision',-1)
                    valid_recall += metirc.get('recall',-1)
                    valid_f1 += metirc.get('f1',-1)
                    valid_Accuracy+=metirc.get('acc',-1)
                    pbar.set_postfix_str(f"Test acc:{metirc.get('acc',-1):.4f} Test f1:{metirc.get('f1',-1):.4f}")
        else :
            for batch in pbar :
                # run batch generation
                pixel_values, labels = batch['pixel_values'].to(device), batch['labels'].to(device)
                outputs = model.generate(pixel_values)
                # compute metrics
                metirc = metricEvaluator.compute_metric(pred_ids=outputs, label_ids=labels,metric_type='multi')
                if metirc['f1']==-1:
                    print('compute error,skip')
                    error_batch+=1
                else :
                    valid_wer +=metirc.get('wer',-1)
                    valid_cer +=metirc.get('cer',-1)
                    valid_precision += metirc.get('precision',-1)
                    valid_recall += metirc.get('recall',-1)
                    valid_f1 += metirc.get('f1',-1)
                    valid_Accuracy+=metirc.get('acc',-1)
                    pbar.set_postfix_str(\
                        f"Test cer:{metirc.get('cer',-1):.4f}Test acc:{metirc.get('acc',-1):.4f} Test f1:{metirc.get('f1',-1):.4f}")


    valid_precision /=(num_eval-error_batch)
    valid_recall /=(num_eval-error_batch)
    valid_f1 /=(num_eval-error_batch)
    valid_cer /=(num_eval-error_batch)
    valid_wer /=(num_eval-error_batch)
    valid_Accuracy /= (num_eval-error_batch)
    print(
          f"Validation CER:{valid_cer}\n"
          f"Validation WER:{valid_wer}\n"
          f"Validation F1:{valid_f1}\n"
          f"Validation Precision:{valid_precision}\n"
          f"Validation Recall:{valid_recall}\n"
          f"Validation WAR:{valid_Accuracy}\n")
    logging.info(
          f"Validation CER:{valid_cer}\n"
          f"Validation WER:{valid_wer}\n"
          f"Validation F1:{valid_f1}\n"
          f"Validation Precision:{valid_precision}\n"
          f"Validation Recall:{valid_recall}\n"
          f"Validation WAR:{valid_Accuracy}\n")
    
    return {
            'precision': valid_precision,
            'recall': valid_recall,
            'f1': valid_f1,
            'cer': valid_cer,
            'wer': valid_wer,
            'acc':valid_Accuracy
        }


#### huggingface trainer
def huggingface_trainer(
        training_args,
        model,
        processor,
        train_dataset,
        val_dataset,
        compute_metrics,
        lr=5e-5
        ):
    # setting optimizer
    # we can use lora+
    lr1 = lr
    lr2 = 2**4*lr1
    groups = []
    for name, param in model.named_parameters():
        if "lora_A" in name:
            groups.append({'params': param, 'lr': lr1})
        elif "lora_B" in name:
            groups.append({'params': param, 'lr': lr1})
    optimizer = AdamW(groups , weight_decay=1e-2)
    # lambda_lr = lambda epoch: 0.1 ** (epoch // 1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor,
        args=training_args,
        optimizers=(optimizer,None),
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    return trainer
    



  