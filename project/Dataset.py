
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob

class BaseDataset(Dataset):
    def __init__(self, processor, max_target_length=20):
        self.processor = processor
        self.max_target_length = max_target_length

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        return pixel_values.squeeze()

    def process_text(self, text):
        try:
            labels = self.processor.tokenizer(text, 
                                              padding="max_length", 
                                              max_length=self.max_target_length).input_ids
            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        except Exception as e:
            print(e)
             # logging.error(f"An error occurred during training: {e}")

        return torch.tensor(labels)
    
### 统一数据集 ####
class MyDataset(BaseDataset):
    '''
    dataset:
        train/
            img_1.jpg\png
            ...
        test/
            img_x.jpg\png
            ...
        train.txt
            train/img_1.jpg\png text
            ...
        test.txt
            ...
        
    '''
    def __init__(self, root_dir, df, processor, max_target_length=20):
        super().__init__(processor, max_target_length)
        # path/dataset/data_name/
        self.root_dir = root_dir
        # columns:img_name text
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]['img_name']
        text = str(self.df.iloc[idx]['text'])
        image_path = os.path.join(self.root_dir, file_name)
        if image_path.endswith('.jp') or image_path.endswith('.pn'):
            image_path =image_path+'g'

        pixel_values = self.process_image(image_path)
        labels = self.process_text(text)
        # return pixel_values, labels
        return {
            'pixel_values': pixel_values,
            'labels': labels,
        }


class BaseDataset2(Dataset):
    def __init__(self, image_processor, tokenizer,max_target_length):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze()
        return pixel_values

    def process_text(self, text):
        try:
            labels = self.tokenizer(text, 
                                    padding="max_length", 
                                    max_length=self.max_target_length, 
                                    truncation=True).input_ids
            labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]
        except Exception as e:
            print(text)
            print(e)
        return torch.tensor(labels)


class SROIEDataset(BaseDataset):
    '''
    dataset:
        images/
            img1.jpg
            ...
        annotations/
            img1.txt
            ...
    '''
    def __init__(self, roots_dir, processor, max_target_length=512):
        super().__init__(processor, max_target_length)
        self.images_dir = os.path.join(roots_dir, 'images')
        self.annotations_dir = os.path.join(roots_dir, 'annotations')
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        label_file = image_file.replace('images', 'annotations').replace('.jpg', '.txt')
        
        pixel_values = self.process_image(image_file)
        
        with open(label_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            text = ' '.join([line.strip() for line in lines])
        
        labels = self.process_text(text)

        # return pixel_values, labels
        return {
            'pixel_values': pixel_values,
            'labels': labels,
        }

class IAMDataset(BaseDataset):
    '''
    dataset:
        train/
            img_1.jpg
            ...
        test/
            img_x.jpg
            ...
        train.txt
        test.txt
    '''
    def __init__(self, root_dir, df, processor, max_target_length=128):
        super().__init__(processor, max_target_length)
        self.root_dir = root_dir
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]['file_name']
        text = self.df.iloc[idx]['text']
        
        image_path = os.path.join(self.root_dir, file_name)
        pixel_values = self.process_image(image_path)
        labels = self.process_text(text)

        # return pixel_values, labels
        return {
            'pixel_values': pixel_values,
            'labels': labels,
        }

class ICDAR2019Task2Dataset(BaseDataset2):
    '''
    dataset:
        train/
            img_1.png
            ...
        test/
            img_x.png
            ...
        train.txt
        test.txt
    '''
    def __init__(self, root_dir, df, image_processor,tokenizer, max_target_length=128):
        super().__init__(image_processor,tokenizer, max_target_length)
        self.root_dir = root_dir
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['image_path'].replace('\\','/')
        text = self.df.iloc[idx]['text']
        # language = self.df.iloc[idx]['language']
        
        pixel_values = self.process_image(image_path)
        labels = self.process_text(text)
        
        # return pixel_values,labels,language
        return {
           'pixel_values': pixel_values,
           'labels': labels,
        }
