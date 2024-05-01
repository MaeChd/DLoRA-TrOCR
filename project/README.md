## Usage

### Training Fine-Tuning (FT)
```sh
python main.py \
    --processor_dir path_to_processor \
    --model_dir path_to_pretrained_model/ \
    --data_dir path_to_datasets \
    --epochs 20 \
    --batch_size 16 \
    --task IAM_SROIE_STR \
    --max_len 64 \
    --lr 1e-5 \
    --eval_mode 
```
### Training Lora
```sh
python main.py \
    --processor_dir path_to_processor \
    --model_dir path_to_pretrained_model/ \
    --data_dir path_to_datasets \
    --epochs 20 \
    --batch_size 16 \
    --task IAM_SROIE_STR \
    --max_len 64 \
    --peft_mode --peft_method LORA \
    --target_modules qkvo --rank 8 --alpha 32 --lr 5e-5 \
    --eval_mode 
```

### Traning DLoRA
```sh
python main.py \
    --processor_dir path_to_processor \
    --model_dir path_to_pretrained_model/ \
    --data_dir path_to_datasets \
    --epochs 20 \
    --batch_size 16 \
    --task IAM_SROIE_STR \
    --max_len 64 \
    --peft_mode --peft_method DLORA \
    --target_modules e_qkv_d_qkvo --rank 8 --alpha 32 --lr 5e-5 \
    --eval_mode 
```

