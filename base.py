


from abc import abstractmethod


# base Layer gpu support
class Base_Layer:

    # objects counter used for state dicts and id asignments
    _id_counter = 0  # Private counter for ID generation

    def __init__(self):
        self.inputs = None
        self.outputs = None

    # this is called when an object is created to increment _id_counter
    def get_next_id(self):
        Base_Layer._id_counter += 1  # Increment counter
        return Base_Layer._id_counter  # Return the new ID

    # base forward method
    @abstractmethod
    def forward(self, inputs):
        # return outputs
        pass

    # base backward method
    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        # return update parameters and returns unput gradient
        pass

    # state dictionary abstract method for saving
    @abstractmethod
    def state_dict(self):
        pass

    # state dictionary abstract method for loading
    @abstractmethod
    def load_state_dict(self):
        pass



