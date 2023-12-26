import math
from neuron import neuron


class layer:

    def __init__(self,num_inputs, num_outputs):
        self.input_size = num_inputs
        self.output_size = num_outputs
        self.neurons=[neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self,inputs):
        self.prev_inputs = inputs
        self.prev_output = [neuron(inputs) for neuron in self.neurons]
        return self.prev_output
    
    def backward(self,derivs):
        #note derivs is a list of derivatives
        for neuron,deriv in zip(self.neurons,derivs):
            neuron.backward(self.prev_inputs,deriv) #use backward on each node
    
    def descend(self, learning_rate):
        for neuron in self.neurons:
            neuron.grad_descend(learning_rate)

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()


class tanhLayer():

    def __init__(self,size):
        self.grad = [0 for _ in range(size)]

    def __call__(self, inputs):
        self.prev_input = inputs
        self.prev_output = [math.tanh(input) for input in inputs]
        return self.prev_output

    def backward(self, derivs):
        pass  

    def zero_grad():
        pass



        
