import os
import re
import torch
import pathlib
import torchvision
import numpy as np
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class GetData:
    def __init__(self) -> None:
        pass

    def Data(self): 
        """
        Info:
          A class to download data from torchvision datasets
        Args:
          None
        Returns:
          This class will return train and test dataset
        """
        train_data = torchvision.datasets.FashionMNIST(
                root="./data",
                train=True,
                download=True,
                transform=ToTensor(),
                target_transform=None
                )

        test_data = torchvision.datasets.FashionMNIST(
                root="./data",
                train=False,
                download=True,
                transform=ToTensor(),
                target_transform=None
                )
        return train_data, test_data


class LoadData:
    def __init__(self, train, test, batch_size: int) -> None:
        self.train = train
        self.test = test
        self.batch_size = batch_size

    def Load(self):
        """
        Info:
          Class to load data in batches
        Args:
          train: Train data set of instance size [1, 28, 28] -> [C, H, W]
          test: Test data set of instance size [1, 28, 28] -> [C, H, W]
        Returns:
          [batch_size,  C, H, W]: Return tensors with batch size i.e if batch_size = 32 then [32, 1, 28, 28]
        """
        train_dataloder = DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=True)
        test_dataloder = DataLoader(dataset=self.test, batch_size=self.batch_size, shuffle=True)
        return train_dataloder, test_dataloder


class ModelV1(nn.Module):
    """
    Info:
      Creating a TinyVGG network from scratch
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer_1 = nn.Linear(in_features=10*7*7, out_features=10)
        self.flattern = nn.Flatten()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
    
    def forward(self, x):
        a = self.conv_1(x)
        a = self.relu(a)
        a = self.relu(a)
        a = self.maxpool(a)
        a = self.conv_2(a)
        a = self.relu(a)
        a = self.conv_2(a)
        a = self.relu(a)
        a = self.maxpool(a)
        a = self.flattern(a)
        a = self.layer_1(a)
        return a

class TrainModel:
    def __init__(self, model, train_dataloder, test_dataloder, epochs, lr) -> None:
        self.model = model
        self.train_dataloder = train_dataloder
        self.test_dataloder = test_dataloder
        self.epochs = epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr = lr)
        self.train_loss = list()
        self.test_loss = list()
        self.train_acc = list()
        self.test_acc = list()

    def Accuracy(self, y_true, y_pred) -> float:
        """
        Info:
          Accuracy function will calculate accuracy for predicted image and actual image
        Args:
          y_true: The real image
          y_pred: The predicted image
        Returns:
          Accuracy
        """
        accuracy = torch.eq(y_true, y_pred).sum().item()
        acc = (accuracy / len(y_true)) * 100
        return acc

    def Train(self) -> None:
        torch.manual_seed(42)
        for epoch in tqdm(range(1, self.epochs + 1)):
            for batch, (x, y) in enumerate(self.train_dataloder):
                self.model.train()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                self.train_loss.append(loss.item())
                #acc = self.Accuracy(y, y_pred)
                #self.train_acc.append(acc)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.model.eval()

            for batch, (x_test, y_test) in enumerate(self.test_dataloder):
                test_pred = self.model(x_test)
                test_loss = self.loss_fn(test_pred, y_test)
                self.test_loss.append(test_loss.item())
                #test_acc = self.Accuracy(y_test, test_pred)
                #self.test_acc.append(test_acc)

            print(f"Epoch: {epoch} | TrainLoss: {loss} |  TestLoss: {test_loss}")




if __name__ == '__main__':
    data = GetData()
    train, test = data.Data()
    dataload = LoadData(train, test, batch_size=32)
    train_dataloder, test_dataloder = dataload.Load()
    model_1 = ModelV1()
    train = TrainModel(model_1, train_dataloder, test_dataloder, epochs=10, lr=0.01)
    train.Train()



