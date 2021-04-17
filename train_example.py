import numpy as np
#from train_ann import *
from neural_net import *
from train_ann import train_ann as train_ANN
from simpleTorch.train_ann import *
x_m = 500  # number of samples
x = np.random.default_rng().uniform(-5, 5, x_m)
y = np.random.default_rng().uniform(-5, 5, x_m)
X = np.array((x, y)).T

F = []
for i in range(0, x_m, 1):
    f = np.array([[100 * (y[i] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2, x[i]*y[i]]])
    F.append(*f.reshape(1,2))
F = np.array(F)
if f.ndim==1:
    F = F.reshape(-1, 1)

model = Model(X.shape[1],F.shape[1])
ANN1 = train_ANN(model, X, F, normalize_y=(0,1), epoch=100, swag=True)
ANN = []
for i in range(20):
    ANN += [train_ANN(model, X, F, normalize_y=(0,1), epoch=100)]
y_s = np.zeros([F.shape[0],F.shape[1],50])
for i in range(20):
    y_s[:,:, i] = ANN[i].predict(X)

y_s1 = y_s.mean(-1)
y_s2 = y_s.std(-1)
# ys = ANN.predict(X)
y_s3 = ANN1.predict_swag(X)



print('2')
