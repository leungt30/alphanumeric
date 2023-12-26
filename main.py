from layer import layer
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
    pass
test4()