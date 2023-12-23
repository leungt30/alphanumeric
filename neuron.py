import random



class neuron:
    
    def __init__(self,num_inputs):
        self.weights = [random.random() for _ in range(num_inputs)]
        self.bias = 0.0
        self.grads = [0 for _ in range(num_inputs + 1)] # bias gradient at the end
    
    def __call__(self, input): #used for forward pass
        return sum([(x*w) for w,x in zip(self.weights,input)]) + self.bias
    
    def __str__(self) -> str:
        return "w: " + str(self.weights) + " b: " + str(self.bias)
    
    def backward(self,gradient_above):
        return 