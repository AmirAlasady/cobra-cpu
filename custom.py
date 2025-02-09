

from sequential import Sequential
from dense_layer import Dense
from activations import *
from losses import *

import matplotlib.pyplot as plt

# base custom model class 
class Model:
    # required forward method
    def forward(self,input_data):
        self.input_data=input_data
        for name, param in self.__dict__.items():       
            if isinstance(param, Dense):
                self.input_data = param.forward(self.input_data)
            if isinstance(param, Sequential):
                self.input_data = param.forward(self.input_data)
            if isinstance(param, Tanh):
                self.input_data = param.forward(self.input_data)
            if isinstance(param, Sigmoid):
                self.input_data = param.forward(self.input_data)
            if isinstance(param, Softmax):
                self.input_data = param.forward(self.input_data)
            if isinstance(param, ReLU):
                self.input_data = param.forward(self.input_data)
        return self.input_data
    
    # required backward method
    def backward(self,grad):
        self.grad=grad
        for name, param in reversed(list(self.__dict__.items())):
            if isinstance(param, Dense):
                self.grad= param.backward(self.grad)
            if isinstance(param, Sequential):
                self.grad= param.backward(self.grad)
            if isinstance(param, Tanh):
                self.grad= param.backward(self.grad)
            if isinstance(param, Sigmoid):
                self.grad= param.backward(self.grad)
            if isinstance(param, Softmax):
                self.grad= param.backward(self.grad)
            if isinstance(param, ReLU):
                self.grad= param.backward(self.grad)
        return self.grad
    
    def train(self, epochs, X, Y, loss, loss_prime, optimizer=None, plot=False):
        self.loss_history = []
        
        # Collect all trainable layers (Dense) including those in Sequential
        trainable_layers = []
        for name, param in self.__dict__.items():
            if isinstance(param, Dense):
                trainable_layers.append(param)
            elif isinstance(param, Sequential):
                # Recursively extract Dense layers from Sequential
                def extract_layers(layer):
                    layers = []
                    if isinstance(layer, Dense):
                        layers.append(layer)
                    elif isinstance(layer, Sequential):
                        for l in layer.layers:
                            layers.extend(extract_layers(l))
                    return layers
                trainable_layers.extend(extract_layers(param))
        
        # Initialize interactive plot
        if plot:
            plt.ion()  # Enable interactive mode
            fig, ax = plt.subplots()
            line, = ax.plot([], [])  # Empty line object
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss During Training')
            ax.grid(True)
            plt.show(block=False)
        
        for i in range(epochs):
            self.error = 0
            for x, y in zip(X, Y):
                self.output = self.forward(x.reshape(-1, 1))
                self.error += loss(y.reshape(-1, 1), self.output)
                self.grad = loss_prime(y.reshape(-1, 1), self.output)
                self.grad = self.backward(self.grad)
                
                if optimizer:
                    optimizer.step(trainable_layers)
            
            self.error /= len(X)
            self.loss_history.append(self.error)
            print(f"{i + 1}/{epochs}, error={self.error}")
            
            if plot:
                # Update the line data
                line.set_xdata(range(len(self.loss_history)))
                line.set_ydata(self.loss_history)
                # Rescale the axes
                ax.relim()
                ax.autoscale_view()
                # Redraw the figure
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)  # Brief pause to update the plot
        
        if plot:
            plt.ioff()  # Turn off interactive mode after training
            plt.show()  # Keep the plot window open                        

    # super state_dict saver 'for all constructor parameters'
    def state_dict(self):
        state_dict  = {}
        for name, param in self.__dict__.items():               
            if isinstance(param, Sequential):  
                state_dict[name] = param.state_dict()
            if isinstance(param, Dense):  
                state_dict[name] = param.state_dict()
            if isinstance(param, Tanh):  
                state_dict[name] = param.state_dict()
            if isinstance(param, Sigmoid):  
                state_dict[name] = param.state_dict()
            if isinstance(param, Softmax):                   
                state_dict[name] = param.state_dict()
            if isinstance(param, ReLU):                   
                state_dict[name] = param.state_dict()
        return state_dict
    
    # super state_dict loader 'for all constructor parameters'
    def load_state_dict(self, state_dict):
        for name, param in self.__dict__.items():
            if isinstance(param, Sequential):  
                param.load_state_dict(state_dict[name])
            if isinstance(param, Dense):  
                param.load_state_dict(state_dict[name])
            if isinstance(param, Tanh):  
                param.load_state_dict()
            if isinstance(param, Sigmoid):  
                param.load_state_dict()
            if isinstance(param, Softmax):  
                param.load_state_dict()
            if isinstance(param, ReLU):  
                param.load_state_dict()