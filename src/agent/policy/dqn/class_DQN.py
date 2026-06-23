import torch.nn as nn
import torch.nn.functional as F
import torch
import util.device as device

class DQN(nn.Module):
    """
    Deep Q Network class

    Parameters:
        inputSize (int): The size of the flattened input state.
        numActions (int): The number of possible actions (left, right, up, down).
        hiddenLayerSize (tuple): The size of the hidden layers. Defaults to (256, 128).

    Attributes:
        fc1 (torch.nn.Linear): The first fully connected layer.
        fc2 (torch.nn.Linear): The second fully connected layer.
        fc3 (torch.nn.Linear): The third fully connected layer.

    Returns:
        torch.nn.Module: A PyTorch neural network module.
    """

    def __init__(self, inputSize: int, numActions: int, hiddenLayerSize: tuple=(256, 128)):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=inputSize, out_features=hiddenLayerSize[0], device=device.accelerator)
        self.fc2 = nn.Linear(in_features=hiddenLayerSize[0], out_features=hiddenLayerSize[1], device=device.accelerator)
        self.fc3 = nn.Linear(in_features=hiddenLayerSize[1], out_features=numActions, device=device.accelerator)


    # @override    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device.accelerator)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


