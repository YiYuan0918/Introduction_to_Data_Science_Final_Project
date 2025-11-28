import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class Synth90kDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=224, img_width=224):
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _load_from_raw_files(self, root_dir, mode):
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()

        paths_file = None
        if mode == 'train':
            paths_file = 'annotation_train.txt'
        elif mode in ('val', 'validation', 'dev'):
            paths_file = 'annotation_val.txt'
        elif mode == 'test':
            paths_file = 'annotation_test.txt'
        else:
            raise ValueError(f"Unsupported split '{mode}'. Expected one of ['train', 'val', 'test'].")

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)

        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        
        try:
            image = Image.open(path).convert('RGB') 
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        #resize
        w, h = image.size
        scale = min(self.img_width/w, self.img_height/h)
        new_w = int(w*scale)
        new_h = int(h*scale)
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)

        #padding
        canvas = Image.new("RGB", (self.img_width, self.img_height), (0, 0, 0))
        left = (self.img_width - new_w)//2
        top = (self.img_height - new_h)//2
        canvas.paste(image, (left, top))

        #normalization
        img_np = np.array(canvas).astype("float32")/255.0
        img_np = (img_np - self.mean) / self.std
        img_np = np.transpose(img_np, (2, 0, 1))    #HWC -> CHW
        image = torch.FloatTensor(img_np)           #shape = [3, 224, 224]

        #character to tensor
        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            
            return image, target, target_length
        else:
            return image


def synth90k_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths

