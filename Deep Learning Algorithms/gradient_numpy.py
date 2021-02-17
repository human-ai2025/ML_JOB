import numpy as np 

# f = w * x
x = np.array([1,2,3,4], dtype = np.float32)
y = np.array([2,4,6,8], dtype = np.float32)

w = 0.0

# model prediction 
def forward(x):
    return w*x


# loss
def loss(y, y_pred):
    return ((y-y_pred)**2).mean()

# gradient 
# MSE = 1/N * (W**X - Y) ** 2
# 

def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred-y).mean()

print(f'Prediction before trainig : f(5) = {forward(5):.3f}')


# Training 
learning_rate = 0.01
n_iter = 100

for epoch in range(n_iter):
    # forward pass
    y_pred = forward(x)

    #loss
    l = loss(y, y_pred)

    # gradients 
    dw = gradient(x, y, y_pred)

    #update weights 
    w -= learning_rate * dw 

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')


