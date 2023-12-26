import random

class neuron:
    
    def __init__(self,num_inputs):
        self.weights = [random.random() for _ in range(num_inputs)]
        self.bias = 0.0
        self.grads = [0 for _ in range(num_inputs + 1)] # bias gradient at the end
    
    def __call__(self, input): #used for forward pass
        return sum([(x*w) for w,x in zip(self.weights,input)]) + self.bias
    
    def __str__(self) -> str:
        return "w: " + str(self.weights) + " b: " + str(self.bias) + " g: " +str(self.grads)
    
    def backward(self,input, gradient_above):
        self.grads = [g+(i*gradient_above) for g,i in zip(self.grads,input)] + [self.grads[-1]+gradient_above]

    def zero_grad(self):
        self.grads = [0 for _ in self.grads]

    def grad_descend(self,learning_rate):
        self.weights=[w-g*learning_rate for w,g in zip(self.weights,self.grads)]
        self.bias = self.bias - self.grads[-1]*learning_rate 

