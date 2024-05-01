import os 
import torch 
import matplotlib.pyplot as plt     
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import AutoTokenizer,AutoImageProcessor
# lora
from peft import (
    get_peft_config, 
    get_peft_model, 
    PeftConfig,
    PeftModel,
    TaskType,
    LoraConfig,
    LoraModel,
    LoKrConfig,
    LoKrModel,
    LoHaConfig,
    LoHaModel,
    LoftQConfig,
    AdaLoraConfig
    # LoftQModel?
    )
from peft import get_peft_model_state_dict, inject_adapter_in_model, set_peft_model_state_dict,LoraConfig


def model_config(model, tokenizer=None, processor=None, beam_search_params=None,if_config = False):
    """Configures model settings, handling tokenizer and beam search parameters."""
    # Determine the tokenizer to use
    tokenizer_to_use = tokenizer if tokenizer is not None else processor.tokenizer

    # Set special tokens
    model.config.decoder_start_token_id = tokenizer_to_use.cls_token_id
    model.config.pad_token_id = tokenizer_to_use.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set beam search parameters from the provided variable, or defaults
    model.config.eos_token_id = tokenizer_to_use.sep_token_id

    if if_config:
        model.config.max_length = beam_search_params.get("max_length", 32)
        model.config.early_stopping = beam_search_params.get("early_stopping", True)
        model.config.no_repeat_ngram_size = beam_search_params.get("no_repeat_ngram_size", 3)#3
        model.config.length_penalty = beam_search_params.get("length_penalty", 2.0)#2.0
        model.config.num_beams = beam_search_params.get("num_beams", 4)#4
    else :
        model.config.max_length = beam_search_params.get("max_length", 20)
        model.config.early_stopping = beam_search_params.get("early_stopping", False)
        model.config.no_repeat_ngram_size = beam_search_params.get("no_repeat_ngram_size", 0)#3
        model.config.length_penalty = beam_search_params.get("length_penalty", 1.0)#2.0
        model.config.num_beams = beam_search_params.get("num_beams", 1)#4



def get_model_and_processor(
    process_dir='./pretrained-models/trocr-base-stage1/',
    model_dir='./pretrained-models/trocr-base-stage1/',
    peft_dict = None,
    model_config_dict = None,
    if_config= True
):
    # Check if directories exist
    if not os.path.exists(process_dir):
        raise FileNotFoundError(f"Processor directory '{process_dir}' not found.")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    peft_mode = peft_dict.get('peft_mode',False)
    mix_model = model_config_dict.get('mix_model',False)
    if mix_model:
        print('use mix model')
        try:
            image_processor = AutoImageProcessor.from_pretrained(process_dir)
            tokenizer = AutoTokenizer.from_pretrained(process_dir)
            pretrained_model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or processor: {e}")
    else :
        print('use trocr pretrained model')
        try:
            processor = TrOCRProcessor.from_pretrained(process_dir)
            pretrained_model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or processor: {e}")

    if peft_mode:
        peft_method = peft_dict.get('peft_method','LORA')
        target_modules =peft_dict.get('target_modules',['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value'])
        r = peft_dict.get('rank',8) # Default value set to 8
        peft_alpha = peft_dict.get('module_alpha',32)  # Default value set to 32
        peft_dropout = peft_dict.get('module_dropout',0.1)  # Default value set to 0.1
        print(f'peft {peft_method} config rank:{r},alpha:{peft_alpha},dropout:{peft_dropout}')
        if peft_method == 'LORA' or peft_method == 'DORA' :
            use_dora =False if peft_method=='LORA' else True
            peft_config= LoraConfig(
                # Assuming these are the desired settings; adjust as necessary.
                target_modules=target_modules,
                inference_mode=False,
                r=r, 
                lora_alpha=peft_alpha, 
                lora_dropout=peft_dropout,
                use_dora=use_dora
            )
        elif peft_method == 'DLORA' or peft_method=='LDORA':
            if peft_method == 'DLORA':
                target_modules_lora = target_modules[-4:]
                target_modules_dora = target_modules[:-4]
            else :
                target_modules_dora = target_modules[-4:]
                target_modules_lora = target_modules[:-4]
            lora_config =  LoraConfig(
                # Assuming these are the desired settings; adjust as necessary.
                target_modules=target_modules_lora,
                inference_mode=False,
                r=r, 
                lora_alpha=peft_alpha, 
                lora_dropout=peft_dropout,
                
            )
            dora_config = LoraConfig(
                # Assuming these are the desired settings; adjust as necessary.
                target_modules=target_modules_dora,
                inference_mode=False,
                r=r, 
                lora_alpha=peft_alpha, 
                lora_dropout=peft_dropout,
                use_dora=True
            )
        elif peft_method == 'LOHA':
            peft_config = LoHaConfig(
                r=r,
                alpha=peft_alpha,
                target_modules=target_modules,
                module_dropout=peft_dropout,
                # modules_to_save=["classifier"],
            )
        elif peft_method == 'LOKR':
            peft_config = LoKrConfig(
                r=r,
                alpha=peft_alpha,
                target_modules=target_modules,
                module_dropout=0.1,
                # modules_to_save=["classifier"],
            )
        elif peft_method =='ADALORA':
            peft_config = AdaLoraConfig(
                    r=r,
                    init_r=12,
                    tinit=200,
                    tfinal=1000,
                    deltaT=10,
                    target_modules=target_modules,
                    # modules_to_save=["classifier"],
                )
        try:
            if peft_method !='DLORA' and  peft_method!='LDORA':
                model = get_peft_model(pretrained_model, peft_config)
                model.print_trainable_parameters()
            else:
                model = inject_adapter_in_model(lora_config, pretrained_model)
                model = inject_adapter_in_model(dora_config, model)
                # now model is vision-language model 
        except Exception as e:
            raise RuntimeError(f"Failed to apply LoRA configuration: {e}")
        print(f'PETF Model OF {peft_method}')
    else:
        model = pretrained_model
        print('fine-tuning model')


    # Customize beam search settings
    beam_search_settings = model_config_dict.get('beam_search_settings',None)

    if mix_model:
        model_config(model,tokenizer,None,beam_search_settings)
        return image_processor,tokenizer,model
    else :
        model_config(model,None,processor,beam_search_settings,if_config)
        return processor, model

def load_model_and_processor(
    process_dir='./pretrained-models/trocr-base-stage1/',
    model_dir='./pretrained-models/trocr-base-stage1/',
    peft_dir = None,
    peft_dict = None,
    model_config_dict = None,
    only_model = False,
    if_config = True
):

    # Check if directories exist
    if not os.path.exists(process_dir):
        raise FileNotFoundError(f"Processor directory '{process_dir}' not found.")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    peft_mode = peft_dict.get('peft_mode',False)
    mix_model = model_config_dict.get('mix_model',False)
    if mix_model:
        print('use mix model')
        try:
            image_processor = AutoImageProcessor.from_pretrained(process_dir)
            tokenizer = AutoTokenizer.from_pretrained(process_dir)
            pretrained_model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or processor: {e}")
    else :
        print('use trocr  model')
        try:
            processor = TrOCRProcessor.from_pretrained(process_dir)
            pretrained_model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or processor: {e}")

    if peft_mode:
        peft_method = peft_dict.get('peft_method','LORA')
        target_modules =peft_dict.get('target_modules',['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value'])
        r = peft_dict.get('rank',8) # Default value set to 8
        peft_alpha = peft_dict.get('module_alpha',32)  # Default value set to 32
        peft_dropout = peft_dict.get('module_dropout',0.1)  # Default value set to 0.1
        print(f'peft {peft_method} config rank:{r},alpha:{peft_alpha},dropout:{peft_dropout}')
        if peft_method!='DLORA' and peft_method!='LDORA':
            try:
                model = PeftModel.from_pretrained(pretrained_model,peft_dir)
            except Exception as e:
                raise RuntimeError(f"Failed to apply peft configuration: {e}")
            # 合并权重
            model.merge_adapter()
        else :
            if peft_method == 'DLORA':
                target_modules_lora = target_modules[-4:]
                target_modules_dora = target_modules[:-4]
            else :
                target_modules_dora = target_modules[-4:]
                target_modules_lora = target_modules[:-4]
        
            lora_config =  LoraConfig(
                # Assuming these are the desired settings; adjust as necessary.
                target_modules=target_modules_lora,
                inference_mode=False,
                r=r, 
                lora_alpha=peft_alpha, 
                lora_dropout=peft_dropout,
                
            )
            dora_config = LoraConfig(
                # Assuming these are the desired settings; adjust as necessary.
                target_modules=target_modules_dora,
                inference_mode=False,
                r=r, 
                lora_alpha=peft_alpha, 
                lora_dropout=peft_dropout,
                use_dora=True
            )
            model = inject_adapter_in_model(lora_config, pretrained_model)
            model = inject_adapter_in_model(dora_config, model)
            from safetensors.torch import load_file,safe_open,load_model
            try:
                model_state_dict = torch.load(os.path.join(peft_dir,'model.safetensors'))
            except:
                # model_state_dict = load_file(os.path.join(peft_dir,'model.safetensors'))
                load_model(model,os.path.join(peft_dir,'model.safetensors'))
            print(model)
        print(f'{peft_method} model to inference')
        # unmerge the LoRA layers from the base model
        # model.unmerge_adapter()
    else:
        model = pretrained_model
        print('fine-tuning model')
    # Customize beam search settings
    beam_search_settings = model_config_dict.get('beam_search_settings',None)
    if only_model:
        if mix_model:
            model_config(model,tokenizer,None,beam_search_settings)
            return model
        else :
            model_config(model,None,processor,beam_search_settings,if_config)
            return model
    else :
        if mix_model:
            model_config(model,tokenizer,None,beam_search_settings)
            return image_processor,tokenizer,model
        else :
            model_config(model,None,processor,beam_search_settings,if_config)
            return processor, model