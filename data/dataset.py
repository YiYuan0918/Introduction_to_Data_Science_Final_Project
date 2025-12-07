import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class Synth90kDataset(Dataset):
    NUM_CLASSES = 88172  # Total number of unique words in lexicon

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=224, img_width=224):
        if root_dir and mode and not paths:
            paths, labels = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            labels = None

        self.paths = paths
        self.labels = labels
        self.img_height = img_height
        self.img_width = img_width

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _load_from_raw_files(self, root_dir, mode):
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
        labels = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                label = int(index_str)
                paths.append(path)
                labels.append(label)

        return paths, labels

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

        # Normalization
        img_np = np.array(canvas).astype("float32") / 255.0
        img_np = (img_np - self.mean) / self.std
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
        image = torch.FloatTensor(img_np)  # shape = [3, img_height, img_width]

        #classification label
        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image


def synth90k_collate_fn(batch):
    """
    Collate function for image classification.

    Args:
        batch: List of tuples (image, label) or list of images

    Returns:
        If labels present: (images, labels) as stacked tensors
        If no labels: images as stacked tensor
    """
    # Check if batch contains labels (tuples) or just images
    if isinstance(batch[0], tuple):
        images, labels = zip(*batch)
        images = torch.stack(images, 0)  # [B, 3, H, W]
        labels = torch.LongTensor(labels)  # [B]
        return images, labels
    else:
        # MAE pretraining case (no labels)
        images = torch.stack(batch, 0)
        return images

