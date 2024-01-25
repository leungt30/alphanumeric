
import pandas as pd
from mlp import mlp
import numpy as np

def label_to_output(label):
    return [1 if i == label else 0 for i in range(10)]

#784 pixels, 10 digit output 
nn = mlp(784,10)

df = pd.read_csv("data/mnist_train.csv")
y_train = df["label"]

upto = 50000
inputs = df.drop("label",axis=1).values[:upto] / 255.0 # preprocess data





y_train = np.array([label_to_output(y) for y in y_train])[:upto]
nn.addSigLayer(100)
nn.addTanhLayer(32)
# nn.addSigLayer(32)
nn.addSigLayer(10)
# nn.addLeakyReluLayer(25,0.1)

# nn.addSoftMaxLayer(10)
nn.build()
# print(inputs[0])
nn.optimize_MSE(inputs,y_train,0.00001,100)

df_test = pd.read_csv("data/mnist_test.csv")
y_test = df_test["label"]
y_test = np.array([label_to_output(y) for y in y_test])
x_test = df_test.drop("label",axis=1).values


mse = 0


for x,y in zip(x_test,y_test):
    output = nn(x)
    mse += np.sum((output - y)**2)


print(mse / len(x_test))

