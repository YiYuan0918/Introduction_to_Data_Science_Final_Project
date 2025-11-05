# Introduction to Data Science Final Project

Final Project of Introduction to Data Science - A deep learning project for computer vision or NLP tasks.

## Table of Contents
- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)

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

