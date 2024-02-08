from cobra import *

# start developing here ...


"""
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


#Layer(2,4)
# -1- single layers model on cpu with save and load 

    # on cpu -------------------------------
l1=Layer(2,3)   # use device='cpu' if wanted 
l2=Tanh()
l3=Layer(3,1)
l4=Tanh()
    

    # saving
l1.state_dict()
    
    # loading
#l1.load_state_dict()


    # forward propagation
array =np.array([[34],[45]])
output=l1.forward(array)

    # back propagation
l1.backward(output,4)





# -2- sequential object on cpu with save and load 


    # on cpu -------------------------------
network=Sequential([
    Layer(2,3),  
    Tanh(),
    Layer(3,1),
    Tanh()
    ])  



    # saving
network.state_dict()
    
    # loading
#network.load_state_dict()



    # forward propagation
output=network.forward(array)

    # back propagation

grad=network.backward(output,3)


network.train(50,X,Y,0.1,Loss_on_cpu.mse,Loss_on_cpu.mse_prime)



# -3- custom models on cpu 

    # on cpu -------------------------------
class Q(Model):
    def __init__(self):
        # layers section
        self.l1=Layer(2,2)
        # sequential networks with compatability with layered objects 
        self.network=Sequential([
            Layer(2,3),
            Tanh(),
            Layer(3,3),
            Tanh()
            ])
        
        self.network2=Sequential([
            Layer(3,3),
            Tanh(),
            Layer(3,1),
            Tanh()
            ])

model1= Q()

    # saving
model1.state_dict()
    
    # loading
#model1.load_state_dict()

    # forward propagation
output=model1.forward(array)

    # back propagation

grad=model1.backward(output,5)
"""