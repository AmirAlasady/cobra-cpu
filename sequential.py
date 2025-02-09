import numpy as np
import matplotlib.pyplot as plt
from dense_layer import Dense
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):  # Removed learning_rate parameter
        for layer in reversed(self.layers):
            grad = layer.backward(grad)     # optimizer.step() will handle the parameters "update" and it just requires initially the 'learning rate'
        return grad

    def train(self, X, Y, epochs,loss_fn, loss_prime_fn ,optimizer=None, verbose=True, plot=False):
        self.loss_history = []
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
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X, Y):
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                
                output = self.forward(x)
                total_loss += loss_fn(y, output)

                grad = loss_prime_fn(y, output)
                self.backward(grad)
                
                # optimizer stepping
                optimizer.step(self.layers)

            avg_loss = total_loss / len(X)
            self.loss_history.append(avg_loss)
            
            if verbose and (epoch % 100 == 0 or epoch == epochs-1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
            
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


    def state_dict(self):
        model_state_dict = {}
        for i, layer in enumerate(self.layers):
            layer_id = layer.id  # Use the unique layer ID for keys
            model_state_dict[layer_id] = layer.state_dict()
        return model_state_dict

    def load_state_dict(self, state_dict):
        keys = state_dict.keys()
        keys=list(keys)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                layer_id = layer.id
                key=keys[i]
                layer.load_state_dict(state_dict[key])
            else:
                pass