import Utils

class NeuralNetwork:
    """
    
    Note: Layer indexes go from left to right (Index: Left 0 to N-1 Right), whereas depth goes from
    right to left (Depth: Left N-1 to 0 Right). So that Neurons in the 0th indexed layer
    have a maximum depth, since they are the farthest away from the final
    layer. The intuition is that you read a Neural Network from left to right,
    and the network can be viewed as a tree that tapers towards the final layer, 
    so that the depth at the final layer should be zero, and that of the leftmost
    hidden layer the maximum depth.
    
    """
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
            # final depth 
            depth = len(sizes) - (x+1)
            max_depth = len(sizes) - 1
            self.neural_network.append(Layer(sizes[x], input_size, depth, max_depth))
        
class Neuron:
    """
    Note: max_depth is N-1 if there are N layers to the ANN.
    """
    def __init__(self, input_size, depth, max_depth, target=None):
        self.input_size = input_size
        self.max_depth = max_depth
        self.depth = depth
        self.target = target # only final Neurons should have a non None target
        self.output = None
        self.input = None # only maximum depth Neurons should have a non None input
        self.weights = []
        self.type = None
        for x in range(0, self.input_size):
            self.weights.append(Utils.generate_random_weight())
        self.type = self.get_type()

    """
    Neuron type is directly related to its depth in relation to the max
    depth of the neural network.
    """
    def get_type(self):
        my_type = 'unknown'
        if (self.depth == 0):
            my_type = 'final'
        elif (self.depth == 1 ):
            my_type = 'output'
        elif (self.depth == None):
            raise ValueError("This neuron's depth cannot be None.")
        elif(self.depth != self.max_depth):
            my_type = 'hidden'
        elif (self.depth == self.max_depth):
            my_type = 'input'
        else:
            my_type = 'unknown'
            raise ValueError("""
                                Could not determine type of this Neuron.
                                 This depth is %d while max depth was %d."""
                                 % (self.depth, self.max_depth))
        return my_type


    """
    Takes an array of inputs and performs a dot product with this neuron's 
    weights. Sets this neuron's output to the result of dot product and returns
    dot product.
    """
    def process_inputs(self, inputs):
        if (len(inputs) != self.input_size):
            raise ValueError("""
                                This Neuron cannot process input array with 
                                length of %d when its input size is %d. """ % (len(inputs), self.input_size))
        output = 0
        for x in range(0, len(inputs)):
            output += self.weights[x] * inputs[x]
        self.output = output
        return output
        

            
            
            
        
        
class Layer:
    def __init__(self, width, input_size, depth, max_depth):
        self.width = width
        self.layer = []
        self.depth = depth
        for x in range (0, self.width):
            self.layer.append(Neuron(input_size, depth, max_depth))
        