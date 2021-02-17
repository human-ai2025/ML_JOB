"""
1. design model (input, output  size, forward pass)
2. construct loss and optimizer 
3. Training Loop 
forward pass: - compute the prediction 
backward pass: - get the gradients 
update the weights 

"""

import torch 
import torch.nn as nn

# f = w * x
x = torch.tensor([[1],
                 [2],
                 [3],
                 [4] ], dtype = torch.float32)

y = torch.tensor([[1],
                 [4],
                 [6],
                 [8] ], dtype = torch.float32)

n_samples, n_features = x.shape
inpSize = n_features
outputSize = n_features


#model = nn.Linear(inpSize, outputSize)

class Linear_regression(nn.Module):
    def __init__(self, inpDim, outDim):
        super(Linear_regression, self).__init__()
        #define the layers 
        self.lin = nn.Linear(inpDim, outDim)

    def forward(self, x):
        return self.lin(x)

model = Linear_regression(inpSize, outputSize)


learning_rate = 0.01
n_iter = 20

x_test = torch.tensor([5], dtype = torch.float32)

# loss
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

print(f'Prediction before trainig : f(5) = {model(x_test).item():.3f}')


# Training 


for epoch in range(n_iter):
    # forward pass
    y_pred = model(x)

    #loss
    l = loss(y, y_pred)

    # gradients 
    l.backward()

    #update weights 
    optimizer.step()

    #zero gradient 
    optimizer.zero_grad()

    if epoch % 1 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(x_test).item():.3f}')
