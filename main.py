import math

import numpy as np
from layer import customActivationLayer, layer, reluLayer, tanhLayer
from mlp import mlp
from neuron import neuron

# inputs = [[1,2,3,4,5],[3,2,1,5,6],[8,8,8,8,8]]
# n2 = neuron(5)


def test1():
    input = [5.0,-3.0]
    n1 = neuron(2)
    expected_output=13.0
    # n1.weights=[1.0,2.0]
    # n1.bias=0.5
    print(n1)


    for i in range(300):
        n1.zero_grad()
        output = n1(input)
        print(output)
        loss = (expected_output-output)**2
        print(loss)
        d = 2*(output-expected_output)
        n1.backward(input,d)
        n1.grad_descend(0.01)
        print(n1)
    print("---------------------")
    print(n1)
    print(n1(input))


def test2():
    #defining desired resutls
    input = [5.0,2.3,0.1,2,-1]
    expected = [-7.0,13.13]

    #defining layer
    l = layer(5,2)
    print("initial: " + str(l(input))) 
    print("initial MSE: " + str(sum((o-e)**2 for o,e in zip(l(input),expected))))

    for _ in range(500):
        
        #forward pass
        out = l(input)

        #calculate loss
        loss = sum((o-e)**2 for o,e in zip(out,expected)) #MSE

        #calculate gradients and descend the gradient
        derivs = (2*(o-e) for o,e in zip(out,expected)) #derivs for MSE
        l.backward(derivs)
        l.descend(0.005)

        #zero gradients
        l.zero_grad()

    print(l(input))
    print(loss)

def test3():
    #defining desired resutls
    inputs = [[5.0,2.3,0.1,2,-1],[5.1,5.3,1.1,2.9,-1]]
    expected = [-7.0,13.13]

    #defining layer
    l = layer(5,2)
    for input in inputs:
        print("initial: " + str(l(input))) 
        print("initial MSE: " + str(sum((o-e)**2 for o,e in zip(l(input),expected))))

    for _ in range(500):
        for input in inputs:
            #forward pass
            out = l(input)

            #calculate loss
            # loss = sum((o-e)**2 for o,e in zip(out,expected)) #MSE

            #calculate gradients
            derivs = (2*(o-e) for o,e in zip(out,expected)) #derivs for MSE
            l.backward(derivs)
        #decend gradient after backward pass of all inputs 
        l.descend(0.005)
        #zero gradients
        l.zero_grad()
    for input in inputs:
        print(l(input))
    # print(loss)

def test4():
    inputs = [[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864],
        [0.15599452, 0.05808361, 0.86617615, 0.60111501, 0.70807258],
        [0.02058449, 0.96990985, 0.83244264, 0.21233911, 0.18182497]]

    expected_outputs = [[0,0],[0.5,0],[1,1]]
    l1 = layer(5,2)
    lt = tanhLayer(2)
    l2 = layer(2,2)
    outputs = [-1,-1,-1]
    
    for _ in range(20000):
        for index,input in enumerate(inputs):
            #forward pass
            outputs[index] = l2(lt(l1(input)))
            
            #loss
            derivs = [2*(o-e) for o,e in zip(outputs[index],expected_outputs[index])]
            

            #backward
            l2.backward(derivs)
            lt.backward(l2.get_grads())
            l1.backward(lt.get_grads())

        #descend
        lr = 0.005
        l2.descend(lr)
        # lt.descend(lr)
        l1.descend(lr)

        #zero grad
        l1.zero_grad()
        lt.zero_grad()
        l2.zero_grad()

    print("Expected outputs: ")
    for eo in expected_outputs:
        print(eo)    
    print("Actual outputs: ")
    for i in inputs:
        print((l2(lt(l1(i))))) 
    print("Error: ")
    for input,eo in zip(inputs,expected_outputs):
        o = (l2(lt(l1(input))))
        print([(x-y)**2 for x,y in zip(eo,o)])
    

def test5():
    inputs = [[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864],
        [0.15599452, 0.05808361, 0.86617615, 0.60111501, 0.70807258],
        [0.02058449, 0.96990985, 0.83244264, 0.21233911, 0.18182497]]

    expected_outputs = [[0,0],[0.5,0],[1,1]]
    l1 = layer(5,2)
    lt = customActivationLayer(2,math.tanh,lambda x: 1 - math.tanh(x)**2)
    l2 = layer(2,2)
    outputs = [-1,-1,-1]
    
    for _ in range(20000):
        for index,input in enumerate(inputs):
            #forward pass
            outputs[index] = l2(lt(l1(input)))
            
            #loss
            derivs = [2*(o-e) for o,e in zip(outputs[index],expected_outputs[index])]
            

            #backward
            l2.backward(derivs)
            lt.backward(l2.get_grads())
            l1.backward(lt.get_grads())

        #descend
        lr = 0.005
        l2.descend(lr)
        # lt.descend(lr)
        l1.descend(lr)

        #zero grad
        l1.zero_grad()
        lt.zero_grad()
        l2.zero_grad()

    print("Expected outputs: ")
    for eo in expected_outputs:
        print(eo)    
    print("Actual outputs: ")
    for i in inputs:
        print((l2(lt(l1(i))))) 
    print("Error: ")
    for input,eo in zip(inputs,expected_outputs):
        o = (l2(lt(l1(input))))
        print([(x-y)**2 for x,y in zip(eo,o)])
        

def test6():
    r = reluLayer(5)
    print(r([1,2,3,4,-5]))
    r.backward([1,2,1,1,1])
    print(r.get_grads())

def test7():
    m = mlp(10,5)
    inputs = [
    [0.25, 0.68, 0.12, 0.93, 0.45, 0.76, 0.21, 0.87, 0.34, 0.59],
    [0.72, 0.18, 0.94, 0.39, 0.67, 0.05, 0.82, 0.29, 0.51, 0.97],
    [0.41, 0.88, 0.15, 0.74, 0.61, 0.33, 0.79, 0.26, 0.69, 0.52],
    [0.98, 0.24, 0.57, 0.81, 0.11, 0.63, 0.45, 0.72, 0.38, 0.19],
    [0.35, 0.77, 0.49, 0.02, 0.68, 0.92, 0.13, 0.56, 0.84, 0.27],
    [0.62, 0.28, 0.76, 0.43, 0.91, 0.09, 0.37, 0.54, 0.18, 0.79],
    [0.85, 0.47, 0.31, 0.66, 0.24, 0.58, 0.73, 0.12, 0.97, 0.41],
    [0.14, 0.89, 0.54, 0.21, 0.68, 0.36, 0.79, 0.02, 0.45, 0.73],
    [0.97, 0.43, 0.78, 0.26, 0.61, 0.15, 0.89, 0.37, 0.54, 0.08],
    [0.64, 0.21, 0.93, 0.57, 0.34, 0.82, 0.46, 0.75, 0.09, 0.68],
    [0.29, 0.67, 0.12, 0.84, 0.51, 0.95, 0.38, 0.62, 0.76, 0.23],
    [0.74, 0.15, 0.49, 0.26, 0.68, 0.04, 0.83, 0.59, 0.31, 0.91],
    [0.37, 0.81, 0.56, 0.02, 0.72, 0.49, 0.14, 0.98, 0.23, 0.65],
    [0.88, 0.42, 0.76, 0.11, 0.59, 0.25, 0.67, 0.34, 0.71, 0.18],
    [0.52, 0.97, 0.28, 0.84, 0.19, 0.63, 0.09, 0.45, 0.76, 0.32],
    [0.25, 0.73, 0.41, 0.68, 0.56, 0.13, 0.87, 0.34, 0.62, 0.79],
    [0.96, 0.17, 0.82, 0.39, 0.75, 0.21, 0.48, 0.63, 0.29, 0.54],
    [0.61, 0.35, 0.74, 0.08, 0.92, 0.46, 0.78, 0.03, 0.69, 0.27],
    [0.84, 0.23, 0.57, 0.69, 0.31, 0.75, 0.42, 0.96, 0.18, 0.62],
    [0.49, 0.78, 0.13, 0.96, 0.24, 0.59, 0.85, 0.37, 0.72, 0.04]
]

    expected_outputs = [
    [0.83, 1.21, 0.57, 0.96, 0.34],
    [0.45, 0.72, 0.09, 1.68, 0.25],
    [0.91, 0.38, 0.76, 0.14, 0.52],
    [1.27, 0.64, 0.18, 0.93, 0.75],
    [0.53, 0.87, 1.41, 0.29, 0.62],
    [0.12, 0.78, 0.36, 0.59, 1.95],
    [0.69, 0.23, 0.81, 0.47, 0.54],
    [0.31, 1.74, 0.58, 0.16, 0.92],
    [0.65, 0.29, 0.43, 0.87, 0.07],
    [1.86, 0.54, 0.19, 0.72, 0.38],
    [0.24, 0.97, 0.71, 0.45, 0.63],
    [0.82, 0.16, 0.37, 0.56, 0.49],
    [0.07, 0.41, 0.68, 0.93, 0.25],
    [0.74, 0.32, 0.58, 0.89, 0.16],
    [0.49, 0.21, 1.96, 0.67, 0.83],
    [0.35, 0.62, 0.78, 0.14, 0.91],
    [0.98, 0.45, 0.27, 0.71, 0.56],
    [0.16, 0.83, 0.54, 0.39, 0.72],
    [1.27, 0.69, 0.84, 0.12, 0.57],
    [0.91, 1.74, 0.43, 0.28, 0.65]
]
    # print(inputs)
    m.addReluLayer(50)
    m.addTanhLayer(50)
    m.addSigLayer(90)
    m.addReluLayer(50)
    m.addTanhLayer(50)
    m.build()

    # print(m(inputs[0]))

    m.optimize_MSE(inputs,expected_outputs,0.00001,10000,0.001)

    for input,output in zip(inputs,expected_outputs):
        
        print([abs(i-o)/o for i,o in zip(m(input),output)])
        # print(sum([(i-o)**2 for i,o in zip(m(input),output)]))

test7()