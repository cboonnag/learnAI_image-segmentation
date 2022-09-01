## example nn.Module 

from re import L
import torch 
from torch import nn

device = 'cpu'

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()                 # module 1
        self.linear_relu_stack = nn.Sequential(     # module 2
            nn.Linear(28*28, 512),                     # submodule 2.1 ... 
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    
    def forward(self, x):
        h = self.flatten(x)
        out = self.linear_relu_stack(h)
        return out 


if __name__ == '__main__':
    net = NeuralNetwork().to(device)
    print(net)

    # [1,28,28]
    x = torch.rand(1,28,28, device = device)

    y_pred = net(x)
    print(y_pred)

    list_of_parameters = net.parameters()
    
    for param in list_of_parameters:
        print(param.shape)

    
    y_target = torch.tensor([[1,0,0,0,0,0,0,0,0,0]], dtype = torch.float32, device = device )

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y_pred, y_target)
    print(loss)

    loss.backward()

    for param in net.parameters():
        print(param.grad.shape) #print gradient