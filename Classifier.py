import torch
import torch.nn as nn


class neural_net(torch.nn.Module):
    def __init__(self):
        super(neural_net, self).__init__()
        self.layer1 = nn.Linear(768*2, 768)
        self.layer2 = nn.Linear(768, 256)
        self.layer3 = nn.Linear(256, 12)
        self.layer4 = nn.Linear(12, 2)
        self.out = nn.Softmax()

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.relu(x)
        x = self.layer4(x)
        x = self.out(x)
        return x
