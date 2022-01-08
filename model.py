from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

        
    def forward(self, x):
        x = x.view(x.shape[0], -1)

        # Hidden layer with ReLU activation
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        # Output layer with log softmax activation
        x = F.log_softmax(self.output(x), dim=1)
        
        return x