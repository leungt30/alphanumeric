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
class tanhLayer():

    def __init__(self,size):
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



        
