from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generate a random synaptic weight
        random.seed(1)

        # Our neuron have 3 input connection and 1 output connection.
        # To do this we assign random weight to a 3x1 matrix, which values ranges from -1 to 1
        # random.random() gives a random value between 0 and 1, the tuple (3,1) specify the size
        # 2 indicate the range (basically random between 0 and 2) and -1 is the off set from 0 to 2 to -1 to 1
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):
        # The sigmoid function, which describe an s shaped curve
        # It'll be used as our activation function
        # We pass the weighted sum of the inputs through this function to normalise them between 0 and 1
        # It'll return us a probability between 0 and 1
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, inputs):
        # Pass the inputs through our neural network (single neuron)
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

    def train(self, training_inputs, training_outputs, iteration_num):
        for iteration in range(iteration_num):
            # Pass the training set through our neural network
            output = self.predict(training_inputs)

            # Calculate the derivative of the cost (error) function: The cost function is (training_output - output) ^ 2
            # Therefore the derivative is 2 * (training_output - output)
            derivative_error = 2 * (training_outputs - output)

            # This calculate the change in cost function in respect to the synaptic weight
            # Training_inputs: the derivative of Z_l in respect to the synaptic weight (Z_l = W_l * training_input)
            # Derivative_error is the derivative of the cost function (training_output - output) ^ 2
            # Sigmoid derivative is the change of the output over the input of a sigmoid function
            gradient = dot(training_inputs.T, derivative_error * self.__sigmoid_derivative(output))

            # Adjust the weight
            self.synaptic_weights += gradient




if __name__ == '__main__':
    neural_network = NeuralNetwork()
    print('Random starting synaptic weights: ')
    print(neural_network.synaptic_weights)

    # The training set with 4 examples, each with 3 input and 1 output
    training_set_inputs = array([[0, 0, 1],
                                 [1, 1, 1],
                                 [1, 0, 1],
                                 [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T  # Transpose the data so they match the input

    # Train the network using the training set 10 000 times and make weight adjustment each time

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print('New synaptic weight after training: ')
    print(neural_network.synaptic_weights)

    # Test the neural network
    print('predicting')
    print(neural_network.predict(array([0, 0, 0])))
