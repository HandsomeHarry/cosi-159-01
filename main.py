import argparse

import torch
import torchvision

from model import Net
from train import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='mnist classification')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    parser.add_argument('--save_dir', type=str, default='./save', help="model save directory")
    parser.add_argument('--load_dir', type=str, default='./save/mnist.pth', help="model load directory")


    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # model
    model = Net()

    # datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=False,
    )

    # trainer
    trainer = Trainer(model=model)
    
    # model inference
    sample = trainer.load_model(path = args.load_dir)  # complete the sample here
    # trainer.infer(sample=sample)

    # model training
    trainer.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="./save/")

    # model evaluation
    trainer.eval(test_loader=test_loader)

    return


if __name__ == "__main__":
    main()
