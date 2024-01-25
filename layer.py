import numpy as np
from neuron import neuron

"""
original implementation of layer. simply stored a bunch of neurons in a list and iterated over them for most functions
"""
class OLDlayer:

    def __init__(self,num_inputs, num_outputs):
        self.input_size = num_inputs
        self.output_size = num_outputs
        self.grads = np.zeros(num_inputs)
        self.neurons=[neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self,inputs:np.ndarray):
        self.prev_inputs = inputs
        self.prev_output = np.array([neuron(inputs) for neuron in self.neurons])
        return self.prev_output
    
    def backward(self,derivs:np.ndarray):
        #note derivs is a list of derivatives

        for neuron,deriv in zip(self.neurons,derivs):
            neuron.backward(self.prev_inputs,deriv) #use backward on each node 

        #calculate gradient with respect to each input. Used to pass gradients through layers
        neuron_grads = np.array([neuron.grads for neuron in self.neurons])
        
        # self.grads = np.array([sum(x) for x in zip(*neuron_grads)]) #SLOW
        self.grads = np.array([sum(x) for x in zip(*neuron_grads)]) #SLOW
        
    
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
using a matrix to store the weights of each neuron instead of holding each neuron in a list.
"""
class layer:

    def __init__(self,num_inputs, num_outputs):
        self.input_size = num_inputs
        self.output_size = num_outputs
        self.grads = np.zeros(num_inputs)
        self.bias_grad = np.zeros(num_outputs)
        self.neurons_grads = np.zeros((num_outputs,num_inputs)) 
        # self.neurons=[neuron(num_inputs) for _ in range(num_outputs)]
        self.neurons_weights = np.random.random((num_outputs,num_inputs)) 
        self.neurons_biases = np.random.random((num_outputs))

    def __call__(self,inputs:np.ndarray):
        self.prev_inputs = inputs
        self.prev_output = np.matmul(self.neurons_weights,inputs) + self.neurons_biases
        return self.prev_output
    
    def backward(self,derivs:np.ndarray):   
        #reminder: this grad is wrt input values, used for back prop on other layers
        self.grads += np.matmul(derivs,self.neurons_weights)

        self.neurons_grads += self.__vector_multiply(derivs,self.prev_inputs)
        
        
        self.bias_grad += derivs

    def __vector_multiply(self,x,y):
        a = x.reshape((-1,1))
        b = y.reshape((1,-1))
        return a@b
        
    
    def descend(self, learning_rate:float):
        self.neurons_weights -= self.neurons_grads*learning_rate
        self.neurons_biases -= self.bias_grad*learning_rate

    def zero_grad(self):
        self.bias_grad.fill(0)
        self.neurons_grads.fill(0)
        self.grads.fill(0)

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
        self.grads = np.zeros(size)

    def __call__(self, inputs:np.ndarray):
        self.prev_input = inputs
        # self.prev_output = [math.tanh(input) for input in inputs]
        self.prev_output = np.tanh(inputs)
        return self.prev_output

    def backward(self, derivs:np.ndarray):
        #since this layer doesn't have any neurons, I keep the gradients as a list and I will pass them on during back prop
        #when combining with a dense layer, we will take the gradients that result from backwards, and use them as derivs for the previous dense layer 
        # self.grads = [current_grad+(1-tanhx**2)*deriv for tanhx,deriv,current_grad in zip(self.prev_output,derivs,self.grads)]
        self.grads = self.grads + (1 - self.prev_output**2) * derivs

    def zero_grad(self):
        self.grads.fill(0)
    
    def get_grads(self):
        return self.grads


    def get_input_size(self):
        return self.size
    
    def get_output_size(self):
        return self.size

#can be used to easily define other activation layers
class customActivationLayer():
    def __init__(self,size,activationFn,activationFnDeriv):
        self.grads = np.zeros(size)
        self.fn = activationFn
        self.dfn = activationFnDeriv
        self.size = size
    
    def __call__(self, inputs:np.ndarray):
        self.prev_input = inputs
        # self.prev_output = np.array([self.fn(input) for input in inputs])
        self.prev_output = self.fn(inputs)
        return self.prev_output
    
    def backward(self, derivs:np.ndarray):
        #since this layer doesn't have any neurons, I keep the gradients as a list and I will pass them on during back prop
        #when combining with a dense layer, we will take the gradients that result from backwards, and use them as derivs for the previous dense layer 
        # self.grads = [current_grad+self.dfn(x)*deriv for x,deriv,current_grad in zip(self.prev_input,derivs,self.grads)]
        self.grads += self.dfn(self.prev_input) * derivs
    
    def zero_grad(self):
        self.grads.fill(0)

    def get_grads(self):
        return self.grads
    
    def get_input_size(self):
        return self.size
    
    def get_output_size(self):
        return self.size

class reluLayer(customActivationLayer):
    def __init__(self,size):

        def single_relu(x):
            return np.maximum(0, x)

        def relu(x:np.ndarray):
            return np.vectorize(single_relu)(x)

        def single_relu_derivative(x):
            return 1 if x > 0.0 else 0.0
        
        def relu_derivative(x:np.ndarray):
            return np.vectorize(single_relu_derivative)(x) #uses np to iterate over all elements
        
        super().__init__(size,relu,relu_derivative)
    
    # def __call__(self, inputs):
    #     return self.layer(inputs)
    
    # def backward(self, derivs):
    #     self.layer(derivs)
    
    # def zero_grad(self):
    #     self.layer.zero_grad()

    # def get_grads(self):
    #     return self.layer.get_grads()

    # def get_output_size(self):
    #     return self.layer.get_output_size()

class sigmoidLayer(customActivationLayer):
    def __init__(self, size):
        def single_sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid(x:np.ndarray):
            return np.vectorize(single_sigmoid)(x)
        
        def sigmoid_derivative(x:np.ndarray):
            sigmoid_x = sigmoid(x)
            return sigmoid_x * (1-sigmoid_x)

        super().__init__(size, sigmoid, sigmoid_derivative)

class leakyReluLayer(customActivationLayer):
    def __init__(self, size, leak):
        self.leak = leak
        def single_leaky_relu(x):
            return x if x > 0.0 else leak * x

        def leaky_relu(x:np.ndarray):
            return np.vectorize(single_leaky_relu)(x)

        def single_leaky_relu_derivative(x):
            return 1 if x > 0.0 else leak 
        
        def leaky_relu_derivative(x:np.ndarray):
            return np.vectorize(single_leaky_relu_derivative)(x)

        super().__init__(size, leaky_relu, leaky_relu_derivative)



class softmaxLayer():
    def __init__(self, size):
        self.grads = np.zeros(size)
        self.size = size

    def __call__(self,input:np.ndarray):
        self.prev_input = input
        pow_input = np.exp(input)
        divisor = np.sum(pow_input)
        
        self.prev_output = pow_input/divisor
        return self.prev_output
        
    def backward(self,derivs:np.ndarray):
        # self.grads = [current_grad+(x*(1-x))*deriv for current_grad,deriv,x  in zip(self.grads,derivs,self.prev_output)]
        self.grads += ((self.prev_output)*(1-self.prev_output)) * derivs

    def zero_grad(self):
        # self.grads = [0 for _ in self.grads]
        self.grads.fill(0.0)

    def get_grads(self):
        return self.grads
    
    def get_input_size(self):
        return self.size
    
    def get_output_size(self):
        return self.size