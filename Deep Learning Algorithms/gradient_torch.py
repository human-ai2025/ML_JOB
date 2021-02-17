import torch

# f = w * x
x = torch.tensor([1,2,3,4], dtype = torch.float32)
y = torch.tensor([2,4,6,8], dtype = torch.float32)

w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)

# model prediction 
def forward(x):
    return w*x


# loss
def loss(y, y_pred):
    return ((y-y_pred)**2).mean()

print(f'Prediction before trainig : f(5) = {forward(5):.3f}')


# Training 
learning_rate = 0.01
n_iter = 10

for epoch in range(n_iter):
    # forward pass
    y_pred = forward(x)

    #loss
    l = loss(y, y_pred)

    # gradients 
    l.backward()

    #update weights 
    with torch.no_grad():
        w -= learning_rate * w.grad

    #zero gradient 
    w.grad.zero_()

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
