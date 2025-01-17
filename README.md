# VAE for 2D Shapes

This project implements a Variational Autoencoder (VAE) for generating 2D shapes, specifically trained on the MNIST dataset.

## Features
- Variational Autoencoder implementation
- MNIST dataset loading and preprocessing
- Training and inference scripts
- Model checkpoint saving
- Generated image visualization

## Requirements
- Python 3.11+
- PyTorch 1.12+
- torchvision
- tqdm

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Training
To try the VAE model in MNIST:
```bash
python train_MNIST.py
```

And train the conformal welding VAE model by provided lesion dataset:
```bash
python train.py
```

Also the `VAE.ipynb` give an example including training, generation and analysis.

## Project Structure
```
.
├── data/                # Dataset and generated outputs
├── models/              # VAE model implementations
│   ├── ConvVAE.py       # Convolutional VAE
│   ├── deepVAE.py       # Deep VAE
│   ├── ResVAE.py        # Residual VAE  
│   ├── SoftIntroVAE_model.py
│   └── VAE.py           # Base VAE implementation
├── utils/               # Utility functions
├── train_MNIST.py       # Main training script
├── train.py             # Generic training script
├── generate.py          # Image generation script
├── read_mhd.py          # Medical image reader
└── requirements.txt     # Python dependencies
```

## License
MIT License
