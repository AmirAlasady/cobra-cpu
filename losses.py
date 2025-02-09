import numpy as np
class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def mse_prime(y_true, y_pred):
        batch_size = y_true.shape[1]
        return 2 * (y_pred - y_true) / batch_size

    @staticmethod
    def cross_entropy(y_true, y_pred, epsilon=1e-12):
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def cross_entropy_prime(y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[1]  # For batch average
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, epsilon=1e-12):
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred))

    @staticmethod
    def categorical_cross_entropy_prime(y_true, y_pred):
        return y_pred - y_true

    @staticmethod
    def accuracy(y_true, y_pred):
        predictions = np.argmax(y_pred, axis=0)
        labels = np.argmax(y_true, axis=0)
        return np.mean(predictions == labels)
