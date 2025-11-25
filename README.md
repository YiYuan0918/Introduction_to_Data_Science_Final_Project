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

### Training

To train the model, configure your parameters in `configs/config.yaml` and run:

```bash
python train.py --config configs/config.yaml
```

### Inference

To run inference on new data:

```bash
python infer.py --model-path <path-to-trained-model> --input <input-data>
```

## Dataset: Synth90k
This project uses the Synth90k synthetic word recognition dataset as the training corpus for ViT-based OCR.

#### Dataset Description
Components contains:
- `Synth90kDataset`：PyTorch Dataset tailored for Synth90k
- `synth90k_collate_fn`：collate function for batching variable-length text labels

#### Preprocessing includes:
- Convert to RGB
- Resize to (224×224)
- Normalize using ImageNet mean/std
- Convert text into character-level label sequence
- Map characters using predefined vocabulary


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
    root_dir= root,
    mode="train",
)

batch = 8
loader = DataLoader(
    dataset,
    batch_size=batch,
    collate_fn = synth90k_collate_fn,
)

for images, targets, lengths in loader:
    pass
```