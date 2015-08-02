import Utils
import math

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
            self.layers.append(Layer(sizes[x], input_size, depth, self.max_depth, self.learning_rate, verbose=self.verbose))
    
    """
    The forward_pass method takes in an input vector and a target vector, 
    feeds it through all neurons, and returns an output vector with 
    the same dimensionality of the target vector.
    """
    def forward_pass(self, input_vector, target_vector):
        self.target_vector = target_vector
        self.input_vector = input_vector # overwritten on each forward pass
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
        for x in range(len(self.layers), 0, -1): # go from final to input layer
            layer = self.layers[x-1]
            plus_one_layer = None
            if layer.layer_type != 'final':
                plus_one_layer = self.layers[layer.depth + 1]
            for neuron in layer.neurons:
                neuron.calculate_my_error(plus_one_layer=plus_one_layer)
        if self.verbose:
            print "Finished backpropogating error"
            
            
    """
    back_propogate_weights function is used after back_propogate_error is called.
    The function will go from final layer to input layer, and update all weights 
    for each neuron.
    """
    def back_propogate_weights(self):
        for x in range(len(self.layers), 0, -1): # go from final to input layer
            layer = self.layers[x-1]
            minus_one_layer = None
            original_input_vector = None
            if layer.layer_type != 'input':
                minus_one_layer = self.layers[layer.depth - 1]
            elif layer.layer_type == 'input':
                original_input_vector = self.input_vector
            else:
                raise ValueError(
                    """
                    The layer type of this layer was %s and was not recognized
                    when back propogating weights (NeuralNetwork.back_propogate_weights)
                    """ % layer.layer_type
                )
            for neuron in layer.neurons:
                neuron.update_my_weights(minus_one_layer=minus_one_layer, original_input_vector=original_input_vector)
        if self.verbose:
            print 'Done updating all weights'
            
    def get_neuron(self, depth, height):
        layer = self.layers[depth]
        neuron = layer.neurons[height]
        return neuron
        
    """
    Get total error with this method. It returns abs value of all errors
    for all output neurons.
    """
    def get_total_output_error(self):
        total_error = 0
        output_layer = self.layers[self.max_depth]
        if output_layer.layer_type != 'final':
            raise ValueError(
                """
                Called get total output error, but layer found had type %s.
                """
                % output_layer.layer_type
            )
        for neuron in output_layer.neurons:
            total_error += abs(neuron.error)
        if self.verbose:
            print "Done calculating total error from output layer."
        return total_error 
        
    def get_network_error(self):
        error = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                error += neuron.error
        return error
        
    def print_outputs(self):
        output_layer = self.layers[self.max_depth]
        print "==="
        print "Outputs for neurons in output layer:"
        for neuron in output_layer.neurons:
            print 'Neuron height: ' + str(neuron.height)
            print 'Neuron output: ' + str(neuron.output)
            print 'Neuron target: ' + str(neuron.target)
            print 'Neuron error: ' + str(neuron.error)
            
                
        
class Neuron:
    """
    Note: max_depth is N-1 if there are N layers to the ANN.
    """
    def __init__(self, input_size, depth, max_depth, learning_rate, target=None, verbose=False, height=None):
        self.input_size = input_size
        self.learning_rate = learning_rate
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
        output = 1.0 / (1.0 + math.exp(-output))
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
        More involved: E = Output * (1-Output) * ( Sum of Each Plus One neuron's error multiplied by its corresponding weight for the self neuron calculating its error)
        
    input:
        No error is calculated for input layer, since input layer weights can be updated
        using only the errors from the +1 layer.
        
    """
    def calculate_my_error(self, plus_one_layer=None):
        if self.verbose:
            print ""
            print '@@@@'
            print 'calculate my error called for neuron:'
            self.print_debug_info()
            print 'Neuron\'s layer type was: ' + self.layer_type
        my_layer_type = self.layer_type
        
        sigmoid_factor = self.output * (1 - self.output)
        error_term = 0
        my_error = 0
        
        if my_layer_type=='final':
            error_term =  (self.target - self.output)
        elif (my_layer_type in ['output','hidden', 'input']):
            if plus_one_layer == None:
                raise ValueError("""Since this neuron is hidden, input, or output type,
                                     it requires a plus one layer, however it was supplied
                                     with None when calculating its error""")
            plus_one_layer.load_error_vector() # get updated values of errors from its neurons
            for neuron in plus_one_layer.neurons:
                error_term += neuron.error * (neuron.weights[self.height])   
        my_error = sigmoid_factor * error_term
        self.error = my_error
        return self.error
        
    """
    The update_my_weights function can be called on this Neuron to update 
    its weights based on its error and the 
    """

    def update_my_weights(self, minus_one_layer=None, original_input_vector=None):
        new_weights = []
        if ((self.layer_type == 'input') & (original_input_vector==None)):
            raise ValueError(
                """Update my weights called on an input neuron, 
                but the original vector passed was None."""
            )
        if ((self.layer_type in ['final', 'output', 'hidden']) & (minus_one_layer==None)):
            raise ValueError(
                """Update my weights called on hidden, output or final 
                neuron but the minus one layer was None. Please call this 
                function on this neuron with a minus_one_layer so it can incorporate
                that's outputs into the new weights.
                """
            )
            
        for idx, weight in enumerate(self.weights):
            if self.layer_type == 'input':
                focus_output = original_input_vector[idx]
            elif self.layer_type in ['final', 'output', 'hidden']:            
                focus_output = minus_one_layer.neurons[idx].output
            else:
                raise ValueError(
                    """In update my weights, this neuron had layer type %s, 
                    and was not a valid type.""" % self.layer_type
                )
            new_weights.append(weight + (self.error * focus_output * self.learning_rate))
        self.weights = new_weights
        return self.weights
        
"""
The Layer is an array of Neurons. The Layer is repsonsible for taking in 
an input vector, and giving an output vector that will be the input vector
for the next layer in front of it.
"""
class Layer:
    def __init__(self, width, input_size, depth, max_depth, learning_rate, verbose=False, target_vector=[]):
        self.width = width
        self.learning_rate = learning_rate
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
            self.neurons.append(Neuron(input_size, depth, max_depth, learning_rate, verbose=self.verbose, height=x))
            
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