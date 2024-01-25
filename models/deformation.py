import torch.nn as nn


class DeformationField(nn.Module):
    """
    The deformation field neural network.
    
    Input:
		Positional encoding for xyz.
  
	Output:
		dx, dy, dz (not positionally encoded!)
    """
    def __init__(self, in_channels=60):
        super(DeformationField, self).__init__()
        self.in_channels = in_channels
        
        self.fc1 = nn.Sequential(nn.Linear(in_channels, 256),
                                 nn.ReLU)
        self.fc2 = nn.Sequential(nn.Linear(256, 256),
                                 nn.ReLU)
        self.fc3 = nn.Sequential(nn.Linear(256, 256),
                                 nn.ReLU)
        self.fc4 = nn.Sequential(nn.Linear(256, 256),
                                 nn.ReLU)
        self.fc5 = nn.Sequential(nn.Linear(256, 256),
                                 nn.ReLU)
        self.fc6 = nn.Sequential(nn.Linear(256, 256),
                                 nn.ReLU)
        self.fc7 = nn.Sequential(nn.Linear(256, 32),
                                 nn.ReLU)
        self.fc8 = nn.Linear(32, 3)
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        
        return x