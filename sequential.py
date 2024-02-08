from dlayer import Layer
import matplotlib.pyplot as plt

class Sequential:
    # list of Layer objects that we will loop on and go forward or backward while saving output per epoch and asign it as input in epoch+1
    def __init__(self,network:list):
        self.network_=network
       
    # forward taking single input and moving forward through the hall network and each output from nth layer is the next input for the next layer "per epoch"
    def forward(self,input_):
        self.output = input_
        for layer in self.network_:
            self.output=layer.forward(self.output)
        return self.output

    # back propagation from the end of the network to the first layer , starting from taking a grad from loss grads then learning rate then back prop fore each layer in reverse 'computing grad of inputs and parameters of nth layer the passing inputs as output grad for next layer for the hall list' 
    def backward(self,grad,lr):
        self.grad=grad
        for layer in self.network_[::-1]:
            self.grad=layer.backward(self.grad,lr)
        return self.grad
    
    def train(self,epochs,X,Y,lr,loss,loss_prime,plot=True):
        self.loss_history=[]
        self.plot=plot
        for i in range(epochs):
            self.error=0
            for x,y in zip(X,Y):
                self.output=self.forward(x)
                self.error += loss(y,self.output)           
                self.grad = loss_prime(y,self.output)
                self.grad=self.backward(self.grad,lr)

            self.error /= len(x)
            self.loss_history.append(self.error)
            print(f"{i + 1}/{epochs}, error={self.error}")
            
        if self.plot:
            plt.plot(self.loss_history)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss During Training')
            plt.show()

    # saving the state dictionary for all layers in the list
    def state_dict(self):
        model_state_dict = {}
        for i, layer in enumerate(self.network_):
            layer_id = layer.id  # Use the unique layer ID for keys
            model_state_dict[layer_id] = layer.state_dict()
        return model_state_dict
    
    # loading the state dictionary for all layers in the list
    def load_state_dict(self, state_dict):       
        keys = state_dict.keys()
        keys=list(keys)
        for i, layer in enumerate(self.network_):
            if isinstance(layer, Layer):
                layer_id = layer.id
                key=keys[i]
                layer.load_state_dict(state_dict[key])