import torch
from torch import nn
from sklearn.model_selection import train_test_split

class DummyDataCreater:
    def __init__(self, weights, bias, start, end, step) -> None:
        self.weights = weights
        self.bias = bias
        self.start = start
        self.end = end
        self.step = step
    
    def CreateData(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        A dummy data creater for Regression
        """
        x = torch.arange(start=self.start, end=self.end, step=self.step).unsqueeze(dim=1)
        y = self.weights * x + self.bias
        return x, y


class ModelV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


class ModelV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=1, out_features=32)
        self.activation_1 = nn.ReLU()
        self.layer_2 = nn.Linear(in_features=32, out_features=64)
        self.activation_2 = nn.ReLU()
        self.layer_3 = nn.Linear(in_features=64, out_features=32)
        self.activation_3 = nn.ReLU()
        self.layer_4 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.layer_4(self.activation_3(self.layer_3(self.activation_2(self.layer_2(self.activation_1(self.layer_1(x)))))))



class TrainModel:
    def __init__(self, model, x_train, y_train, x_test, y_test, epochs, lr) -> None:
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.loss_fn = nn.L1Loss()
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)

    def Accuracy(self, y_true, y_pred):
        accuracy = torch.eq(y_true, y_pred).sum().item()
        acc = (accuracy / len(y_pred)) * 100
        return acc

    def Train(self):
        torch.manual_seed(42)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            y_pred = self.model(self.x_train)
            loss = self.loss_fn(y_pred, self.y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            if epoch % 10 == 0:
                with torch.inference_mode():
                    test_pred = self.model(self.y_test)
                    test_loss = self.loss_fn(test_pred, self.y_test)
                    print(f"Epoch: {epoch} | Loss: {test_loss} | Weights and Bias: {self.model.state_dict()}")



data = DummyDataCreater(weights=0.7, bias=0.3, start=0, end=1, step=0.02)
x, y = data.CreateData()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model_1 = ModelV1()

train_1 = TrainModel(model_1, x_train, y_train, x_test, y_test, epochs=250, lr=0.05)
train_1.Train()

#print(100 * "#")

#model_2 = ModelV2()

#train_2 = TrainModel(model_2, x_train, y_train, x_test, y_test, epochs=150, lr=0.01)
#train_2.Train()
