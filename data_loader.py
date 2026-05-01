"""
Data Loading Utilities for Res_AttnGAN
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CUBDataset(Dataset):
    """
    CUB (Caltech-UCSD Birds) Dataset for text-to-image synthesis
    
    Expected directory structure:
    data/
        CUB_200_2011/
            images/
            text_c10/
            filenames.pickle
    """
    
    def __init__(
        self,
        data_path='data/CUB_200_2011',
        split='train',
        transform=None,
        img_size=64
    ):
        super(CUBDataset, self).__init__()
        
        self.data_path = data_path
        self.split = split
        self.img_size = img_size
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Load filenames
        filenames_path = os.path.join(data_path, 'filenames.pickle')
        if not os.path.exists(filenames_path):
            print(f"Warning: {filenames_path} not found")
            self.filenames = []
        else:
            with open(filenames_path, 'rb') as f:
                filenames = pickle.load(f)
                self.filenames = filenames[split]
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Load image
        img_path = os.path.join(
            self.data_path, 'images', f"{filename}.jpg"
        )
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except:
            print(f"Error loading image: {img_path}")
            # Return random image as fallback
            image = torch.randn(3, self.img_size, self.img_size)
        
        # Load text
        text_path = os.path.join(
            self.data_path, 'text_c10', f"{filename}.txt"
        )
        try:
            with open(text_path, 'r') as f:
                captions = [line.strip() for line in f.readlines()]
                caption = np.random.choice(captions)
        except:
            caption = "unknown bird"
        
        return {
            'image': image,
            'caption': caption,
            'filename': filename
        }


class COCODataset(Dataset):
    """
    COCO Dataset for text-to-image synthesis
    """
    
    def __init__(
        self,
        data_path='data/coco',
        split='train',
        transform=None,
        img_size=64,
        num_captions_per_image=5
    ):
        super(COCODataset, self).__init__()
        
        self.data_path = data_path
        self.split = split
        self.img_size = img_size
        self.num_captions = num_captions_per_image
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return 30000  # Placeholder
    
    def __getitem__(self, idx):
        # Placeholder implementation
        image = torch.randn(3, self.img_size, self.img_size)
        caption = f"image {idx}"
        
        return {
            'image': image,
            'caption': caption,
            'idx': idx
        }


class DummyDataset(Dataset):
    """
    Dummy Dataset for testing
    Generates random images and captions
    """
    
    def __init__(self, num_samples=1000, img_size=64, vocab_size=5000):
        super(DummyDataset, self).__init__()
        
        self.num_samples = num_samples
        self.img_size = img_size
        self.vocab_size = vocab_size
        
        # Create dummy captions
        self.captions = [
            f"a {np.random.choice(['red', 'blue', 'green', 'yellow'])} "
            f"{np.random.choice(['bird', 'flower', 'cat', 'dog'])}"
            for _ in range(num_samples)
        ]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random image
        image = torch.randn(3, self.img_size, self.img_size)
        
        # Random tokens
        tokens = torch.randint(0, self.vocab_size, (10,))  # Fixed length 10
        length = torch.tensor(np.random.randint(5, 11))
        
        caption = self.captions[idx]
        
        return {
            'image': image,
            'caption': caption,
            'tokens': tokens,
            'length': length
        }


class TextTokenizer:
    """
    Simple text tokenizer for captions
    """
    
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
    def tokenize(self, caption, max_length=20):
        """
        Tokenize a caption string
        
        Args:
            caption: Text caption
            max_length: Maximum length of tokens
            
        Returns:
            tokens: Token indices
            length: Actual length before padding
        """
        words = caption.lower().split()
        tokens = [1]  # START token
        
        for word in words[:max_length-2]:
            # Simple: hash word to get index
            if word not in self.word2idx:
                idx = (hash(word) % (self.vocab_size - 4)) + 4
                self.word2idx[word] = idx
                self.idx2word[idx] = word
            tokens.append(self.word2idx[word])
        
        tokens.append(2)  # END token
        
        length = len(tokens)
        
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(0)  # PAD token
        
        return torch.tensor(tokens[:max_length]), torch.tensor(length)


def get_data_loader(
    dataset_name='dummy',
    data_path=None,
    split='train',
    batch_size=32,
    num_workers=4,
    img_size=64,
    shuffle=True
):
    """
    Create data loader for training/testing
    
    Args:
        dataset_name: Name of dataset ('dummy', 'cub', 'coco')
        data_path: Path to data directory
        split: Data split ('train' or 'test')
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Image size
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader object
    """
    
    if dataset_name == 'dummy':
        dataset = DummyDataset(
            num_samples=1000,
            img_size=img_size,
            vocab_size=5000
        )
    elif dataset_name == 'cub':
        if data_path is None:
            data_path = 'data/CUB_200_2011'
        dataset = CUBDataset(
            data_path=data_path,
            split=split,
            img_size=img_size
        )
    elif dataset_name == 'coco':
        if data_path is None:
            data_path = 'data/coco'
        dataset = COCODataset(
            data_path=data_path,
            split=split,
            img_size=img_size
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )
    
    return dataloader


# Collate function for custom batching
def default_collate_fn(batch):
    """
    Default collate function for batching
    """
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    
    return {
        'images': images,
        'captions': captions
    }
