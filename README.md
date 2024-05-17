# DLoRA-TrOCR
[Paper:](https://arxiv.org/abs/2404.12734)

## Architecture:
In the Transformer layer of the image encoder, **DoRA** weight was embedded into the multi-head attention weight part. In the Transformer layer of the text decoder, the **LoRA** weights are partially embedded for the multi-head attention weights and the masked multi-head attention weights.
The overall architecture of the method is as follows,

![架构](https://github.com/VceChang/DLoRA-TrOCR/assets/87841835/b8932d51-4645-41af-98f2-6dc2e7bc6213)



## Usage:
### Dataset setting
```python
Folder:
Dataset/
    IAM/
        train/
            img1.jpg
            ...
        gt_train.txt
        test/
            ...
        gt_test.txt
    SROIE/
        train/
            img1.jpg
            ...
        test/
            ...
    STR_BENCHMARKS/
        IC13/
            train/
                img1.jpg
                ...
            test/
                ...
        IC15/
            train/
                img1.jpg
                ...
            test/
                ...
        III5K/
            train/
                img1.jpg
                ...
            test/
                ...
        SVT
            train/
                img1.jpg
                ...
            test/
                ...
        SVTP
            test/
                ...
        CU80
            test/
                ...
            
```

### train
```shell
# use huggingface trainer
python train.py --processor_dir path to processor  \
--model_dir path to model --save_dir path to save \
--data_dir path to datasets --log_file_dir ./log_file/ \
--epochs 20 --batch_size 16 --task IAM_SROIE_STR --max_len 32 \
--peft_mode --peft_method LORA \
--target_modules e_qkv --rank 8 --alpha 32 --lr 5e-5 \
--eval_mode --use_trainer

# use lora
python train.py --processor_dir path to processor  \
--model_dir path to model --save_dir path to save \
--data_dir path to datasets --log_file_dir ./log_file/ \
--epochs 20 --batch_size 16 --task IAM_SROIE_STR --max_len 32 \
--peft_mode --peft_method LORA \
--target_modules e_qkv --rank 8 --alpha 32 --lr 5e-5 \
--eval_mode
```

### test
```shell
python main.py --processor_dir path to processor  \
--model_dir path to model --save_dir path to save \
--data_dir path to datasets --log_file_dir ./log_file/ \
--epochs 20 --batch_size 16 --task IAM_SROIE_STR --max_len 32 \
--peft_mode --peft_method LORA \
--target_modules e_qkv --rank 8 --alpha 32 --lr 5e-5 \
--eval_mode --only_test
```


