import numpy as np
from base_layer import Base_Layer
class Tanh(Base_Layer):
    def __init__(self):
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"
    def forward(self, inputs):
        self.inputs = inputs
        return np.tanh(inputs)

    def backward(self, output_gradient):
        return output_gradient * (1 - np.tanh(self.inputs)**2)
    
    def state_dict(self):
        return {"activation": type(self).__name__}

    def load_state_dict(self, state_dict):
        pass
    #-----
    def parameters(self):
        return []

class ReLU(Base_Layer):
    def __init__(self):
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, output_gradient):
        return output_gradient * (self.inputs > 0).astype(float)

    def state_dict(self):
        return {"activation": type(self).__name__}
    
    def load_state_dict(self, state_dict):
        pass
    #-----
    def parameters(self):
        return []
    
class Sigmoid(Base_Layer):
    def __init__(self):
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, output_gradient, ):
        return output_gradient * (self.outputs * (1 - self.outputs))

    def state_dict(self):
        return {"activation": type(self).__name__}
    
    def load_state_dict(self, state_dict):
        pass
    #-----
    def parameters(self):
        return []
    
class Softmax(Base_Layer):

    def __init__(self):
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"

    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs))
        self.outputs = exp / np.sum(exp, axis=0)
        return self.outputs

    def backward(self, output_gradient):
        return output_gradient  # Assumes combined with cross-entropy loss

    def state_dict(self):
        return {"activation": type(self).__name__}
    
    def load_state_dict(self, state_dict):
        pass
    #-----
    def parameters(self):
        return []