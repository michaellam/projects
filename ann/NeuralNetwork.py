import Utils

class NeuralNetwork:
    def __init__(self, input_size, learning_rate=1, sizes=[2,1], minimum_error=0.10):
        if (input_size == None):
            raise ValueError("Input size cannot be None type.")
        self.neural_network = []
        self.learning_rate = learning_rate
        self.sizes = sizes
        self.input_size = input_size
        self.minimum_error = minimum_error
        for x in range(0, len(sizes)):
            if (x==0):
                input_size = self.input_size
            else:
                input_size = self.sizes[x-1]
            self.neural_network.append(Layer(sizes[x], input_size))
        
class Neuron:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = []
        for x in range(0, self.input_size):
            self.weights.append(Utils.generate_random_weight())
        
class Layer:
    def __init__(self, width, input_size):
        self.width = width
        self.layer = []
        for x in range (0, self.width):
            self.layer.append(Neuron(input_size))
        