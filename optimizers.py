import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, layers):
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}

    def step(self, layers):
        for layer in layers:
            for param, grad in layer.parameters():
                param_id = id(param)
                
                if param_id not in self.velocities:
                    self.velocities[param_id] = np.zeros_like(param)
                
                self.velocities[param_id] = self.momentum * self.velocities[param_id] + grad
                param -= self.learning_rate * self.velocities[param_id]

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0  # Time step starts at 0

    def step(self, layers):
        self.t += 1
        for layer in layers:
            for param, grad in layer.parameters():
                param_id = id(param)
                if param_id not in self.m:
                    self.m[param_id] = np.zeros_like(grad)
                    self.v[param_id] = np.zeros_like(grad)
                
                # Update biased moment estimates
                self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
                self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)
                
                # Compute bias-corrected estimates
                m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
                v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
                
                # Update parameters
                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, gamma=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.cache = {}

    def step(self, layers):
        for layer in layers:
            for param, grad in layer.parameters():
                param_id = id(param)
                
                if param_id not in self.cache:
                    self.cache[param_id] = np.zeros_like(grad)
                
                self.cache[param_id] = self.gamma * self.cache[param_id] + (1 - self.gamma) * (grad ** 2)
                param -= self.learning_rate * grad / (np.sqrt(self.cache[param_id]) + self.epsilon)
