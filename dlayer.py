


from base import Base_Layer
import numpy as np


# Layer class gpu support
class Layer(Base_Layer):

    def __init__(self, input_size, output_size, name=None):

      self.weights = np.random.randn(output_size, input_size)
      self.bias = np.random.randn(output_size, 1)
      # Assign ID for state_dict usage
      self.id = f"{self.__class__.__name__}_{super().get_next_id()}"
      self.name = name

    def forward(self, inputs):
      self.inputs = inputs
      return np.dot(self.weights, self.inputs) + self.bias

    def backward(self, output_gradient, learning_rate):
      weights_gradient = np.dot(output_gradient, self.inputs.T)
      input_gradient = np.dot(self.weights.T, output_gradient)
      self.weights -= learning_rate * weights_gradient
      self.bias -= learning_rate * output_gradient
      return input_gradient

    def state_dict(self):
      return {
            "weights": self.weights,
            "bias": self.bias,
      }
    
    def load_state_dict(self, state_dict):
      self.weights = state_dict['weights']
      self.bias = state_dict['bias']

        


