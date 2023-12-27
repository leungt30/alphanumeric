import math
from neuron import neuron


class layer:

    def __init__(self,num_inputs, num_outputs):
        self.input_size = num_inputs
        self.output_size = num_outputs
        self.grads = [0 for _ in range(num_inputs)]
        self.neurons=[neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self,inputs):
        self.prev_inputs = inputs
        self.prev_output = [neuron(inputs) for neuron in self.neurons]
        return self.prev_output
    
    def backward(self,derivs):
        #note derivs is a list of derivatives

        for neuron,deriv in zip(self.neurons,derivs):
            neuron.backward(self.prev_inputs,deriv) #use backward on each node 

        #calculate gradient with respect to each input. Used to pass gradients through layers
        neuron_grads = [neuron.grads for neuron in self.neurons]
        self.grads = [sum(x) for x in zip(*neuron_grads)]
        
    
    def descend(self, learning_rate):
        for neuron in self.neurons:
            neuron.grad_descend(learning_rate)

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()

    def get_grads(self):
        return self.grads
    
    def get_input_size(self):
        return self.input_size
    
    def get_output_size(self):
        return self.output_size

    
"""
Activation layer using tanh as the activation function. 

I implemented this before the custom activation class that was based on this.
This is technically redudant since I can use the custom to create tanh but I will keep it for now
"""    
class tanhLayer():

    def __init__(self,size):
        self.size = size
        self.grads = [0 for _ in range(size)]

    def __call__(self, inputs):
        self.prev_input = inputs
        self.prev_output = [math.tanh(input) for input in inputs]
        return self.prev_output

    def backward(self, derivs):
        #since this layer doesn't have any neurons, I keep the gradients as a list and I will pass them on during back prop
        #when combining with a dense layer, we will take the gradients that result from backwards, and use them as derivs for the previous dense layer 
        self.grads = [current_grad+(1-tanhx**2)*deriv for tanhx,deriv,current_grad in zip(self.prev_output,derivs,self.grads)]
        
    def zero_grad(self):
        self.grads = [0 for _ in self.grads]
    
    def get_grads(self):
        return self.grads


    def get_input_size(self):
        return self.size
    
    def get_output_size(self):
        return self.size

#can be used to easily define other activation layers
class customActivationLayer():
    def __init__(self,size,activationFn,activationFnDeriv):
        self.grads = [0 for _ in range(size)]
        self.fn = activationFn
        self.dfn = activationFnDeriv
        self.size = size
    
    def __call__(self, inputs):
        self.prev_input = inputs
        self.prev_output = [self.fn(input) for input in inputs]
        return self.prev_output
    
    def backward(self, derivs):
        #since this layer doesn't have any neurons, I keep the gradients as a list and I will pass them on during back prop
        #when combining with a dense layer, we will take the gradients that result from backwards, and use them as derivs for the previous dense layer 
        self.grads = [current_grad+self.dfn(x)*deriv for x,deriv,current_grad in zip(self.prev_input,derivs,self.grads)]
    
    def zero_grad(self):
        self.grads = [0 for _ in self.grads]

    def get_grads(self):
        return self.grads
    
    def get_input_size(self):
        return self.size
    
    def get_output_size(self):
        return self.size

class reluLayer():
    def __init__(self,size):
        def relu(x):
            return max(0, x)

        def relu_derivative(x):
            return 1 if x > 0 else 0
        
        self.layer = customActivationLayer(size,relu,relu_derivative)
    
    def __call__(self, inputs):
        return self.layer(inputs)
    
    def backward(self, derivs):
        self.layer(derivs)
    
    def zero_grad(self):
        self.layer.zero_grad()

    def get_grads(self):
        return self.layer.get_grads()

    def get_output_size(self):
        return self.layer.get_output_size()