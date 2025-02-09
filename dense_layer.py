import numpy as np
from base_layer import Base_Layer


class Dense(Base_Layer):
    def __init__(self, input_size, output_size, name=None, initialization='xavier'):
        super().__init__()
        self.name = name
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"
        if initialization == 'xavier':
            std = np.sqrt(2.0 / (input_size + output_size))
            # dirctly working shaping the layer as output size and input size thus no need for trenspose
            self.weights = np.random.randn(output_size, input_size) * std
        else:
            self.weights = np.random.randn(output_size, input_size) * 0.01
            
        self.bias = np.zeros((output_size, 1))
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.weights, self.inputs) + self.bias

    def backward(self, output_gradient):
        batch_size = output_gradient.shape[1]
        self.weights_grad = np.dot(output_gradient, self.inputs.T) / batch_size
        self.bias_grad = np.mean(output_gradient, axis=1, keepdims=True)
        return np.dot(self.weights.T, output_gradient)

    def parameters(self):
        return [
            (self.weights, self.weights_grad),
            (self.bias, self.bias_grad)
        ]

    def state_dict(self):
        return {
            "weights": self.weights,
            "bias": self.bias,
        }
    
    def load_state_dict(self, state_dict):
        self.weights = state_dict['weights']
        self.bias = state_dict['bias']
