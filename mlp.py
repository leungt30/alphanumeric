from typing import List
import numpy as np

from tqdm import tqdm
from layer import customActivationLayer, layer, leakyReluLayer, reluLayer, sigmoidLayer, softmaxLayer, tanhLayer


class mlp:
    def __init__(self,num_inputs,num_outputs):
        self.layers : List[layer] =[layer(num_inputs,num_inputs)]
        # self.layerSize=[] #refers to the number of outputs of each hidden layer
        # self.functions = []
        self.input_size = num_inputs
        self.output_size = num_outputs
    
    
    def addLayer(self,size:int, activationFn, activationFnDeriv):
        input_size = self.layers[-1].get_output_size()
        newLayer = layer(input_size,size)
        newActivationLayer = customActivationLayer(size, activationFn,activationFnDeriv)
        self.layers.extend([newLayer,newActivationLayer])

    def addReluLayer(self,size:int):
        input_size = self.layers[-1].get_output_size()
        newLayer = layer(input_size,size)
        newActivationLayer = reluLayer(size)
        self.layers.extend([newLayer,newActivationLayer])

    def addLeakyReluLayer(self,size:int,leak):
        input_size = self.layers[-1].get_output_size()
        newLayer = layer(input_size,size)
        newActivationLayer = leakyReluLayer(size,leak)
        self.layers.extend([newLayer,newActivationLayer])

    def addTanhLayer(self,size:int):
        input_size = self.layers[-1].get_output_size()
        newLayer = layer(input_size,size)
        newActivationLayer = tanhLayer(size)
        self.layers.extend([newLayer,newActivationLayer])

    def addSigLayer(self,size:int):
        input_size = self.layers[-1].get_output_size()
        newLayer = layer(input_size,size)
        newActivationLayer = sigmoidLayer(size)
        self.layers.extend([newLayer,newActivationLayer])

    def addSoftMaxLayer(self,size:int):
        input_size = self.layers[-1].get_output_size()
        newLayer = layer(input_size,size)
        newActivationLayer = softmaxLayer(size)
        self.layers.extend([newLayer,newActivationLayer])

    def build(self):
        #add the final output layer. This layer's input count must match the layer before's output count
        self.layers.append(layer(self.layers[-1].get_output_size(),self.output_size))
    

    def backward(self,derivs):
        current_grad = derivs
        for layer in reversed(self.layers):
            layer.backward(current_grad)
            current_grad = layer.get_grads()
        
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def descend(self,learning_rate):
        for theLayer in self.layers:
            if (isinstance(theLayer,layer)):
                theLayer.descend(learning_rate)


    def optimize_MSE(self,inputs,expected_outputs,learning_rate,iterations):
        
        for i in range(iterations):
            # if i%1000 == 0:
            print(i)
            mse = float(0)
            for index,input in enumerate(pbar:=tqdm(inputs)):
                
                #forward pass
                output = self(input)
                #loss using MSE 
                # mse += sum([(e-o)**2 for o,e in zip(output,expected_outputs[index])])
                curr_loss =  np.sum(np.power(expected_outputs[index] - output,2))
                # loss_derivs = [2*(o-e) for o,e in zip(output,expected_outputs[index])]
                loss_derivs = 2 * (output * expected_outputs[index])

                if index%10 == 0:
                    # pbar.set_description(f"Current Loss: {curr_loss} | Label: {expected_outputs[index]} | Pred: {output}")
                    pbar.set_description(f"Current Loss: {curr_loss}")
                
                #backward pass
                self.zero_grad()
                self.backward(loss_derivs)

                self.descend(learning_rate)



    def __call__(self,input):
        output=input
        for l in self.layers:
            output=l(output)
        return output
