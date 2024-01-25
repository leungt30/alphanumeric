import numpy as np
from layer import layer, sigmoidLayer
from mlp import mlp
def loss_info(output,expected_output):
    loss =  (output - expected_output)**2
    loss_grad = 2 * (output - expected_output)
    return loss,loss_grad

# sig = sigmoidLayer(5)
# lay = layer(4,5)
nn = mlp(4,5)
nn.addSigLayer(100)
nn.build()
# print(lay.grads)
# print(lay.neurons_weights)

inputs = np.array([[1,2,4,5]])
outputs = np.array([[3,6,9,12,15]])

nn.optimize_MSE(inputs,outputs,0.001,1000)

print(nn(inputs[0]))
print((nn(inputs[0])-outputs[0])**2)