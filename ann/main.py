import NeuralNetwork


# example of a network learning a simple pattern
input_vector = [1,0,1,0]
target_vector = [1,0]
nn = NeuralNetwork.NeuralNetwork(4, learning_rate=1, sizes=[3, 2], minimum_error=0.10, verbose=True)

errors = []
for x in range (0, 100):
    nn.forward_pass(input_vector, target_vector)
    nn.back_propogate_error()
    nn.back_propogate_weights()
    errors.append(nn.get_total_output_error())
print errors
