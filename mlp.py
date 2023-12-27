from layer import customActivationLayer, layer, reluLayer, tanhLayer


class mlp:
    def __init__(self,num_inputs,num_outputs):
        self.layers=[layer(num_inputs,num_inputs)]
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

    def addTanhLayer(self,size:int):
        input_size = self.layers[-1].get_output_size()
        newLayer = layer(input_size,size)
        newActivationLayer = tanhLayer(size)
        self.layers.extend([newLayer,newActivationLayer])

    def build(self):
        #add the final output layer. This layer's input count must match the layer before's output count
        self.layers.append(layer(self.layers[-1].get_output_size(),self.output_size))
    
    def __call__(self,input):
        output=input
        for l in self.layers:
            output=l(output)
        return output
