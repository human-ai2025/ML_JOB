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
x = torch.tensor([1,2,3,4], dtype = torch.float32)
y = torch.tensor([2,4,6,8], dtype = torch.float32)

w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)

# model prediction 
def forward(x):
    return w*x

learning_rate = 0.01
n_iter = 10

# loss
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr = learning_rate)

print(f'Prediction before trainig : f(5) = {forward(5):.3f}')


# Training 


for epoch in range(n_iter):
    # forward pass
    y_pred = forward(x)

    #loss
    l = loss(y, y_pred)

    # gradients 
    l.backward()

    #update weights 
    optimizer.step()

    #zero gradient 
    optimizer.zero_grad()

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
