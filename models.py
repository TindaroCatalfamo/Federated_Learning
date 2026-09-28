import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        
        # primo layer convoluzionale
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # secondol ayer convoluzionale
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # layer di pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # fully connected layer
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        
        # output layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class MNISTCNN_Attention(nn.Module):
    
    def __init__(self):
        super(MNISTCNN_Attention, self).__init__()
        
        # primo layer convoluzionale
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # secondo layer convoluzionale
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # layer di pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # fully connected layer
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        
        # output layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, return_attention=False):
        x = self.pool(F.relu(self.conv1(x)))
        act2 = F.relu(self.conv2(x))  
        x = self.pool(act2)           
        
        feat = x.view(-1, 32 * 7 * 7)
        x_fc = F.relu(self.fc1(feat))
        logits = self.fc2(x_fc)
        
        if return_attention:
            att_map = torch.sum(act2 ** 2, dim=1)  
            return logits, att_map
            
        return logits

