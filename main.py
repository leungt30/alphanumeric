
import pandas as pd
from mlp import mlp
import numpy as np
from tqdm import tqdm
def label_to_output(label):
    return [1 if i == label else 0 for i in range(10)]

#784 pixels, 10 digit output 
nn = mlp(784,10)

df = pd.read_csv("data/mnist_train.csv")
y_train = df["label"]
inputs = df.drop("label",axis=1).values



y_train = np.array([label_to_output(y) for y in y_train])
nn.addSigLayer(50)
nn.addReluLayer(25)
nn.addTanhLayer(50)
nn.addSoftMaxLayer(10)
nn.build()

nn.optimize_MSE(inputs,y_train,0.001,100)

df_test = pd.read_csv("data/mnist_test.csv")
y_test = df_test["label"]
y_test = np.array([label_to_output(y) for y in y_test])
x_test = df_test.drop("label",axis=1).values


mse = 0


for x,y in zip(x_test,y_test):
    output = nn(x)
    mse += (output - y)**2


print(mse / len(x_test))

