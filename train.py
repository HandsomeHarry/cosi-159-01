import os
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from utils import AverageMeter


class Trainer:
    """ Trainer for MNIST classification """

    def __init__(self, model: nn.Module):
        self._model = model

    def train(
            self,
            train_loader: DataLoader,
            epochs: int,
            lr: float,
            save_dir: str,
    ) -> None:
        """ Model training, TODO: consider adding model evaluation into the training loop """

        optimizer = optim.SGD(params=self._model.parameters(), lr=lr)
        loss_track = AverageMeter()
        self._model.train()

        print("Start training...")
        for i in range(epochs):
            tik = time.time()
            loss_track.reset()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self._model(data)

                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                loss_track.update(loss.item(), n=data.size(0))

            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.5f" % (i + 1, epochs, elapse, loss_track.avg))
            # self.eval(test_loader=train_loader)

        print("Training completed, saving model to %s" % save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), os.path.join(save_dir, "mnist.pth"))

        return

    def eval(self, test_loader: DataLoader) -> float:
        """ Model evaluation, return the model accuracy over test set """
        self.load_model(path="./save/mnist.pth") #Load the model from the .pth file.
        self._model.eval()

        loss_track = AverageMeter()

        for data, target in test_loader:
            output = self._model(data)  # forward pass
            loss = F.nll_loss(output, target)   # compute loss
            loss.backward() # backward pass

            loss_track.update(loss.item(), n=data.size(0))  # update loss

        print ("Evaluation completed, accuracy: %.5f" % (1 - loss_track.avg)) #Accuracy = 1 - loss
        return 1 - loss_track.avg

    def infer(self, sample: Tensor) -> int:
        """ Model inference: input an image, return its class index """
        self.load_model(path="./save/mnist.pth") # Load the model from the .pth file.
        self._model.eval()  # set model to evaluation mode

        output = self._model(sample)    # forward pass
        print("Inference completed, class index: %d" % output.argmax(dim=1))    # return the class index
        return output.argmax(dim=1) # return the class index

    def load_model(self, path: str) -> None:
        """ load model from a .pth file """
        self._model.load_state_dict(torch.load(path))
        return None

