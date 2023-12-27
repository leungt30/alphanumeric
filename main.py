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
    inputs = [np.random.rand(10) for _ in range(20)]
    # expected_outputs = [np.random.rand(5) for _ in range(20)]

    m.addReluLayer(21)
    m.addTanhLayer(15)
    m.addReluLayer(25)

    m.build()

    print(m(inputs[0]))

test7()