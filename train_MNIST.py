
import torch
import torchvision.datasets as datasets
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models.VAE import VariationalAutoEncoder

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 128
LR_RATE = 3e-4

MNIST_DIR = "data/mnist"

# Dataset Loading with custom resources

# Set custom MNIST resources
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms


def load_mnist(batch_size=BATCH_SIZE, data_dir=MNIST_DIR):
    # Set custom MNIST resources
    resources = [
        ('train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
        ('train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
        ('t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
        ('t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
    ]

    # Override default resources
    datasets.MNIST.resources = resources
    datasets.MNIST.mirrors = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
        'http://yann.lecun.com/exdb/mnist/'
    ]

    # Load dataset
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    return dataset

def train(model, dataset):
    # Initialize model and optimizer
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
    loss_fn = nn.BCELoss(reduction="sum")

    # Start Training
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader))
        for i, (x, _) in loop:
            # Forward Pass
            x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
            x_reconstructed, mu, sigma = model(x)

            # Compute loss
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # Backprop
            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

def inference(model, dataset, digit, num_examples=1):
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"{MNIST_DIR}/outputs/{digit}_{example}.png")


if __name__ == "__main__":
    dataset = load_mnist()
    model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
    
    train(model, dataset)

    for idx in range(10):
        inference(model.to('cpu'), dataset, idx, num_examples=5)
