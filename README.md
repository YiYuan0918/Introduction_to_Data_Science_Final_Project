# Introduction to Data Science Final Project

Final Project of Introduction to Data Science - A deep learning project for computer vision or NLP tasks.

## Table of Contents
- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [dataset](#dataset-synth90k)

## File Structure

```
Introduction_to_Data_Science_Final_Project/
├── README.md                  # Project documentation
├── CONTRIBUTING.md            # Cooperation and contribution guidelines
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
│
├── configs/                   # Configuration files
│   └── config.yaml           # Training and model configurations
│
├── data/                      # Data directory (not tracked by git)
|   ├── __init__.py           # data package initialization
|   ├── dataset.py            # dataset & dataloader implementation 
│   └── .gitkeep              # Keep empty directory in git
│
├── models/                    # Model definitions
│   └── __init__.py           # Model package initialization
│
├── utils/                     # Utility functions
│   └── __init__.py           # Utils package initialization
│
├── train.py                   # Training script
├── infer.py                   # Inference/prediction script
│
└── tests/                     # Unit tests
    └── .gitkeep              # Keep empty directory in git
```

## Installation

### Prerequisites
- Python 3.11 or higher
- (Optional) CUDA-capable GPU for faster training

### Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd Introduction_to_Data_Science_Final_Project
```

2. Create a virtual environment:
```bash
python -m venv .venv

# On Linux/Mac:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

This project uses a two-stage training approach: (1) MAE pretraining for self-supervised visual representation learning, and (2) classifier fine-tuning for word-level image classification. The MAE stage is optional but recommended for improved performance.

### Configuration Setup

Example configuration files are provided as templates. Copy them to create your own configs:

```bash
cp configs/mae.ex.yaml configs/mae.yaml
cp configs/cls.ex.yaml configs/cls.yaml
```

Edit the config files to adjust parameters like data paths, batch size, learning rate, epochs, etc. See the example configs for all available parameters.

### Stage 1: MAE Pretraining (Optional)

The MAE (Masked Autoencoder) stage pretrains a Vision Transformer encoder by learning to reconstruct masked image patches. This provides strong visual representations for downstream tasks.

```bash
python train.py --config configs/mae.yaml
```

The pretrained model will be saved to `outputs/mae_pretrain`. You can skip this stage and use the base pretrained weights from HuggingFace if preferred.

### Stage 2: Classifier Training

The classifier stage fine-tunes the model for word-level image classification using CrossEntropyLoss on the Synth90k dataset. It loads the MAE pretrained encoder and adds a classification head for 88,172 word classes.

```bash
python train.py --config configs/cls.yaml
```

The trained classifier will be saved to `outputs/classifier`. Make sure the `mae_checkpoint_for_init` parameter in your config points to the correct MAE checkpoint path from Stage 1.

**Note:** With 88,172 classes, the classification head is large (~67M parameters). You may need to reduce batch size (to 32 or 16) if you encounter out-of-memory errors.

### Resuming Training

If training is interrupted, you can resume from a checkpoint:

```bash
python train.py --config <config_file> --resume_from <checkpoint_path>
```

### Inference

To run inference on new data:

```bash
python infer.py --model-path <path-to-trained-model> --input <input-data>
```

## Dataset: Synth90k
This project uses the Synth90k synthetic word recognition dataset for word-level image classification.

#### Dataset Description
Components contains:
- `Synth90kDataset`：PyTorch Dataset tailored for Synth90k with 88,172 word classes
- `synth90k_collate_fn`：collate function for batching images and class labels

#### Preprocessing includes:
- Convert to RGB
- Resize to (224×224) with aspect-ratio preserving padding
- Normalize using ImageNet mean/std
- Map images to word class labels (0-88171) from lexicon.txt


### Synth90k dataset Structure 
After downloading and extracting the dataset, the directory typically follows this structure:
```
mnt/
└── ramdisk/
    └── max/
        └── 90kDICT32px/
            ├── annotation.txt       # all samples: <relative_path> <lexicon_id>
            ├── annotation_test.txt  # test split
            ├── annotation_train.txt # train split
            ├── annotation_val.txt   # validation split
            ├── imlist.txt           # list of all images relative paths
            ├── lexicon.txt          # each line is a word label; index = <lexicon_id>
            │  
            ├── 1/
            │   ├── 1/
            │   │   ├── 1_pontifically_58805.jpg
            │   │   ├── 2_Senoritas_69404.jpg
            |   |   └── ...
            |   |
            |   ├── 2/
            |   ├── 3/
            |   └── ...
            |  
            ├── 2/
            ├── 3/
            └── ...
```

### Example Usage (DataLoader)
```python
from torch.utils.data import DataLoader
from data import Synth90kDataset, synth90k_collate_fn

root = r"mnt/ramdisk/max/90kDICT32px"

dataset = Synth90kDataset(
    root_dir=root,
    mode="train",
)

batch = 8
loader = DataLoader(
    dataset,
    batch_size=batch,
    collate_fn=synth90k_collate_fn,
)

for images, labels in loader:
    # images: [batch_size, 3, 224, 224]
    # labels: [batch_size] with values in range [0, 88171]
    pass
```