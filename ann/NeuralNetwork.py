import Utils

class NeuralNetwork:
    """
    Note: Layer indexes go from left to right (Index: Left 0 to N-1 Right), and 
    depth also goes from left to right. So that input layers should have zero depth
    values and output layers have max depth.
    """
    def __init__(self, input_size, learning_rate=1, sizes=[2,1], minimum_error=0.10, verbose=False):
        if (input_size == None):
            raise ValueError("Input size cannot be None type.")
        self.layers = []
        self.learning_rate = learning_rate
        self.sizes = sizes
        self.input_size = input_size
        self.minimum_error = minimum_error
        self.max_depth = len(sizes) - 1
        self.verbose = verbose
        self.target_vector = None # only set when doing a forward pass
        for x in range(0, len(sizes)):
            if (x==0):
                input_size = self.input_size
            else:
                input_size = self.sizes[x-1]
            # final depth goes from left to right
            depth = x
            self.layers.append(Layer(sizes[x], input_size, depth, self.max_depth, verbose=self.verbose))
    
    """
    The forward_pass method takes in an input vector and a target vector, 
    feeds it through all neurons, and returns an output vector with 
    the same dimensionality of the target vector.
    """
    def forward_pass(self, input_vector, target_vector):
        self.target_vector = target_vector
        output_layer_neurons = self.layers[len(self.layers)-1].neurons
        if len(target_vector) != len(output_layer_neurons):
            raise ValueError(
                    """
                    The length of the target vector (%d) must be equal
                    to the length of the output layer (%d).
                    """ 
                    % (len(target_vector), len(output_layer_neurons))
            )
        if self.verbose:
            print "Feeding forward now. With this original input vector:"
            print input_vector

        original_input = input_vector
        for idx, layer in enumerate(self.layers):
            if self.verbose:
                print "processing layer number %d" % idx            
            if get_my_layer_type(layer.depth, self.max_depth) == "input":
                if self.verbose:
                    print "I was input layer"
                layer.process_input_vector(original_input)
            else:
                if self.verbose:
                    print "I was not input layer"
                previous_layer_output = self.layers[idx - 1].output_vector
                layer.process_input_vector(previous_layer_output, target_vector = self.target_vector)
        if self.verbose:
            print "Done feeding input through forward pass"
        return self.layers[len(self.layers)-1].output_vector
        
    """
    Back propogate method should be called after a forward pass if the 
    error from the final output layer and target vector is not sufficiently 
    small.
    
    This method goes backwards layer by layer, and calculates the error
    for each applicable Neuron.
    
    After this method is called, all error is current in the Network, and
    a method update_weights should then be called before doing another 
    forward pass.
    """
    def back_propogate_error(self):
        """Pseudo code"""
        """
        for layer in self.layers:
            for neuron in layer.neurons:
                if type final:
                    calculate error with target
                if type hidden or output:
                    calculate error with +1 layer error vector
                if type input:
                    return None
        """
        for layer in self.layers:
            plus_one_layer = None
            if layer.layer_type != 'final':
                plus_one_layer = self.layers[layer.depth + 1]
            for neuron in layer.neurons:
                neuron.calculate_my_error(plus_one_layer=plus_one_layer)
        if self.verbose:
            print "Finished backpropogating error"
                
        
class Neuron:
    """
    Note: max_depth is N-1 if there are N layers to the ANN.
    """
    def __init__(self, input_size, depth, max_depth, target=None, verbose=False, height=None):
        self.input_size = input_size
        self.max_depth = max_depth
        self.depth = depth
        self.target = target # only final Neurons should have a non None target
        self.output = None # the dot product of current input vector and current weights, overwritten on each forward pass
        self.input_vector = [] # overwritten on forward pass
        self.weights = []
        self.layer_type = self.get_my_layer_type()
        self.verbose = verbose
        self.error = None
        self.height = height
        for x in range(0, self.input_size):
            self.weights.append(Utils.generate_random_weight())
        self.layer_type = get_my_layer_type(self.depth, self.max_depth)

    def get_my_layer_type(self):
        return get_my_layer_type(self.depth, self.max_depth)

    """
    Returns info about this Neuron for debugging purposes
    """
    def print_debug_info(self):
        print '================='
        print '== Im a Neuron =='
        print 'My type: ' + self.layer_type
        print 'My depth: ' + str(self.depth)
        print 'My height: ' + str(self.height)
        print 'My output: ' + str(self.output)
        print 'My target: ' + str(self.target)
        print 'My Error: ' + str(self.error)
        print 'My weights:' + ', '.join(str(x) for x in self.weights)
        print 'My input vector: ' + ', '.join(str(x) for x in self.input_vector)
        print '== That is All =='
        print '================='

    """
    Takes an array of inputs and performs a dot product with this neuron's 
    weights. Sets this neuron's output to the result of dot product and returns
    dot product.
    """
    def process_input_vector(self, input_vector):
        self.input_vector = input_vector
        if (len(input_vector) != self.input_size):
            self.print_debug_info()
            raise ValueError(   """
                                This Neuron cannot process input array with 
                                length of %d when its input size is %d. 
                                """  % 
                                ( len(input_vector), self.input_size )
                            )
        output = 0
        for x in range(0, len(input_vector)):
            output += self.weights[x] * input_vector[x]
        self.output = output
        if self.verbose:
            print 'Processing input vector: '
            self.print_debug_info()
        return output
        
        
    """
    The calculate_my_error function calculates and then sets an error value 
    on the self object. 
    
    The error is determined differently depending on the neuron type.
    
    If layer type is:
    final:
        Straightforward: E = Output * (1-Output) * (Target - Output)
        
    output || hidden:
        More involved: E = Output * (1-Output) * ( Error Vector of + 1 layer dotted with Weights Vector of this layer)
        
    input:
        No error is calculated for input layer, since input layer weights can be updated
        using only the errors from the +1 layer.
        
    """
    def calculate_my_error(self, plus_one_layer=None):
        my_layer_type = self.layer_type
        error = 0
        if my_layer_type=='final':
            error = self.output * (1 - self.output) * (self.target - self.output)
        elif (my_layer_type=='output') or (my_layer_type=='hidden'):
            if self.verbose:
                print 'Calculating internal error of hidden or output level neuron.'
            if plus_one_layer == None:
                raise ValueError("""Since this neuron is hidden or output type,
                                     it requires a plus one layer, however it was supplied
                                     with None when calculating its error""")
            plus_one_layer.load_error_vector() # get updated values of errors from its neurons
            error_vector = plus_one_layer.error_vector
            my_weights_dot_error = 0
            for x in range(0, len(self.weights)):
                print 'foo'
                
            
        else:
            print 'bar'
            return # input layer does not have error vector
        self.error = error
            
        

"""
The Layer is an array of Neurons. The Layer is repsonsible for taking in 
an input vector, and giving an output vector that will be the input vector
for the next layer in front of it.
"""
class Layer:
    def __init__(self, width, input_size, depth, max_depth, verbose=False, target_vector=[]):
        self.width = width
        self.neurons = []
        self.depth = depth
        self.max_depth = max_depth
        self.output_vector = [] # overwritten each pass
        self.input_vector = [] # overwritten each pass
        self.verbose = verbose
        self.error_vector = []
        self.target_vector = target_vector
        self.layer_type = self.get_my_layer_type()
        for x in range (0, self.width):
            self.neurons.append(Neuron(input_size, depth, max_depth, verbose=self.verbose, height=x))
            
    def get_my_layer_type(self):
        return get_my_layer_type(self.depth, self.max_depth)        
        
    def process_input_vector(self, input_vector, target_vector = None):
        self.output_vector = []
        self.input_vector = input_vector
        if self.verbose:
            print 'Processing input vector: '
            print input_vector
        for idx, neuron in enumerate(self.neurons):
            self.output_vector.append(neuron.process_input_vector(input_vector))
            if self.verbose:
                print "Determining if final neuron for target attribution."
            if neuron.layer_type == 'final':
                my_target = target_vector[idx]
                neuron.target = my_target
                if self.verbose:
                    print "This was a final neuron, so target was assigned of %d." % my_target
            else:
                if self.verbose:
                    print "This was not a final neuron, so no target was assigned."
            if self.verbose:
                print "Info on this neuron:"
                neuron.print_debug_info()
            
    def load_error_vector(self):
        self.error_vector = []
        for neuron in self.neurons:
            self.error_vector.append(neuron.error)

"""
Layer type is directly related to its depth in relation to the max
depth of the neural network. A Neuron and Layer could potentially require
needing to know which type of layer it is in when making decisions.
"""
def get_my_layer_type(my_depth, max_depth, verbose=False):
    net_has_hidden_layers = max_depth > 1 # zero indexed, needs at least 3 layers to have hidden layer
    if verbose:
        print "Get my layer called."
        print "my depth: " + str(my_depth)
        print "max depth: " + str(max_depth)
    my_type = 'unknown'
    if (my_depth == max_depth):
        my_type = 'final'
    elif ( (my_depth == max_depth - 1) & net_has_hidden_layers):
        my_type = 'output'
    elif(my_depth == 0):
        my_type = 'input'
    else:
        my_type = 'hidden'

    if (my_type=='unknown'):
        raise ValueError("""
                            Could not determine type of this Layer.
                             This layer's depth is %d while max depth was %d."""
                             % (my_depth, max_depth))    
    if verbose:
        print "layer type was determined as: " + my_type
    return my_type    