from base import Base_Layer
import numpy as np


class Activation(Base_Layer):
    def __init__(self):
        pass
    
    def get_next_id2(self):
        return super().get_next_id()

    def forward(self):
        pass

    def backward(self):
        pass
    

class Tanh(Activation):
   def __init__(self):
       self.id = f"{self.__class__.__name__}_{super().get_next_id2()}"

   def forward(self, input_):    
        self.inp=input_   
        return np.tanh(self.inp)

   def backward(self, output_gradient, learning_rate):

        temp = 1 - np.tanh(self.inp) ** 2   
        return np.multiply(output_gradient,temp) 

   def state_dict(self):
        return {"activation": type(self).__name__,}
           
   def load_state_dict(self):
       pass 
   

class Sigmoid(Activation):

    def __init__(self):
        self.id = f"{self.__class__.__name__}_{super().get_next_id2()}"
             
        
    def forward(self, input_):

        self.inp=input_   
        self.outp= 1 / (1 + np.exp(-self.inp))
        return self.outp

    def backward(self, output_gradient, learning_rate):
        
        
        self.a_tmp= 1 / (1 + np.exp(-self.inp)) 
        self.b_tmp= self.a_tmp * (1-self.a_tmp)
        return np.multiply(output_gradient,self.b_tmp)

    def state_dict(self):
        return {"activation": type(self).__name__,}
           
    def load_state_dict(self):
        pass
    


class Softmax(Base_Layer):

    def __init__(self):
        self.id = f"{self.__class__.__name__}_{super().get_next_id()}"    
 
    def forward(self, input_):
        self.inp = np.exp(input_)
        self.outp = self.inp / np.sum(self.inp)
        return self.outp
       
    def backward(self, output_gradient, learning_rate):     
        self.n = np.size(self.outp)
        return np.dot((np.identity(self.n) - self.outp.T) * self.outp, output_gradient)
    
    def state_dict(self):
        return {"activation": type(self).__name__,}
           
    def load_state_dict(self):
        pass
    

class ReLU(Activation):
   def __init__(self, ):
       super().__init__()
       self.id = f"{self.__class__.__name__}_{super().get_next_id2()}"
   
   def forward(self, input_):      
        self.inp = input_
        return np.maximum(self.inp, 0)

   def backward(self, output_gradient, learning_rate):
        temp = self.inp > 0
        return np.multiply(output_gradient, temp)

   def state_dict(self):
        return {"activation": type(self).__name__,}
           
   def load_state_dict(self):
       pass 
