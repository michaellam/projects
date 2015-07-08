import NeuralNetwork as ann

net = ann.NeuralNetwork(3, sizes=[2,2,1])

print net.neural_network[0].layer[0].type
