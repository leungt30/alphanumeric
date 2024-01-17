import random

import numpy as np

class neuron:
    
    def __init__(self,num_inputs):
        self.weights = np.random.rand(num_inputs) #initialize with random weights
        self.bias = 0.0
        self.grads = np.array([0.0 for _ in range(num_inputs)]) 
        self.bias_grad = 0.0
    
    def __call__(self, input): #used for forward pass
        # return sum([(x*w) for w,x in zip(self.weights,input)]) + self.bias
        return np.sum(input*self.weights) + self.bias
    
    def __str__(self) -> str:
        return "w: " + str(self.weights) + " b: " + str(self.bias) + " g: " +str(self.grads)
    
    def backward(self,input:np.ndarray, gradient_above):
        # self.grads = [g+(i*gradient_above) for g,i in zip(self.grads,input)]+ ([self.grads[-1]+gradient_above])
        
        self.grads += input * gradient_above
        self.bias_grad += gradient_above
  




    def zero_grad(self):
        self.grads.fill(0.0)

    def grad_descend(self,learning_rate):
        # self.weights=np.array([w-g*learning_rate for w,g in zip(self.weights,self.grads)])
        self.weights -= self.grads - learning_rate
        self.bias -= self.bias_grad*learning_rate 

