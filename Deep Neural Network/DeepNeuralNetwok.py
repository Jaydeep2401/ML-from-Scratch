import numpy as np

class DeepNeuralNetwork:
    def __init__(self):
        self.parameters = None


    def initialize_parameters(self, layers_dims, layers_activations):
        # Intializing W and b
        np.random.seed(3)
        parameters = {}     # Dictionary Parameters contaning W and b of each layer
        L = len(layers_dims)

        for l in range(1, L):
            # If the activation for layer l is ReLU we intialize weight with He method else Xavier method
            if layers_activations[l] == "relu":
                parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
                
            else:
                parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(1 / layers_dims[l - 1])

            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        return parameters


    def linear_activation_forward(self, A_prev, W, b, activation):
        # Calculates Linear function and its activation for a given layer
        Z = np.dot(W, A_prev) + b
        linear_cache = A_prev, W, b

        if activation == "sigmoid":
            A = 1 / (1 + np.exp(-Z))

        elif activation == "relu":
            A = np.where(Z <= 0, 0, Z)

        elif activation == "tanh":
            A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

        activation_cache = Z

        # cache contains A, W, b & Z for current layer
        cache = (linear_cache, activation_cache)

        return A, cache

  
    def forward_propagation(self, X, parameters, layers_activations):
        # Implements forward prop for the architecture
        # caches stores parameters contains A, W, b & Z for all layers. To be used during back prop
        caches = []
        A = X
        L = len(parameters) // 2

        # Forward propagation
        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation = layers_activations[l])
            caches.append(cache)

        # Constraining last layer as Sigmoid since using Network for binary classification
        AL, cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")

        caches.append(cache)

        # AL = Activations of last layer and caches has linear_cache and activation_cache of each layer, that will be used during backprop.
        return AL, caches 


    
    def compute_cost(self, AL, Y, parameters, lambd):
        # Cross Entropy Loss Function with L2 regularization
        m = Y.shape[1]
        l2_norm = 0
        L = len(parameters) // 2

        # Frobenius norm of W
        for l in range(1, L + 1):
            l2_norm += np.sum(np.square(parameters["W" + str(l)]))

        L2_regularization_cost = lambd / (2 * m) * l2_norm
        cross_entropy = - (np.dot(Y, np.log(AL.T)) + np.dot((1 - Y), np.log(1 - AL.T))) / m
        
        # Adding penalty to cross entropy loss
        cost = cross_entropy + L2_regularization_cost
        cost = np.squeeze(cost)
        
        return cost

  
    def linear_activation_backward(self, dA, cache, activation, lambd):
        # Calculates gradients for given layer
        linear_cache, activation_cache = cache

        A_prev, W, b = linear_cache
        Z = activation_cache

        m = A_prev.shape[1]
        
        # Calculating dZ according to the activations mentioned
        if activation == "relu":
            dZ = dA * np.where(Z > 0, 1, 0)      
            
        elif activation == "sigmoid":
            dZ = dA * (np.exp(-Z) / ((1 + np.exp(-Z)) ** 2)) 

        elif activation == "tanh":
            dZ = dA * (1 - (((np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))) ** 2))    
        
        dA_prev = np.dot(W.T, dZ)
        dW = (np.dot(dZ, A_prev.T) + lambd * W) / m
        db = np.sum(dZ, axis = 1, keepdims = True) / m 

        return dA_prev, dW, db


    def backward_propagation(self, AL, Y, caches, lambd, layers_activations):
        # Implements backward prop for the architecture
    
        grads = {}      # Dictionary of grads conating gradients of all the layers
        L = len(caches) # the number of layers
        Y = Y.reshape(AL.shape) # Y is reshaped as AL
        
        # Calculating Gradients for last layer seperately
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, caches[L - 1], "sigmoid", lambd)
        grads["dA" + str(L - 1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp
        
        # Calculating Gradients for layers l = L - 2 to l = 1
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA_prev_temp, current_cache, layers_activations[l + 1], lambd)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads


    def update_parameters(self, params, grads, learning_rate):
        parameters = params.copy()
        L = len(parameters) // 2 

        # Updating Parameters for each layer
        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
            
        return parameters


    def model(self, X, Y, layers_dims, layers_activations, learning_rate = 0.0075, num_iterations = 3000, print_cost = False, lambd = 0):
        #Constructing L-Layer Neural network according to given layer_dims and layer_activations
        np.random.seed(1)

        self.parameters = self.initialize_parameters(layers_dims, layers_activations)

        X = X.T
        Y = Y.T

        # Loop (gradient descent)
        for i in range(0, num_iterations):
            # Forward propagation
            AL, caches = self.forward_propagation(X, self.parameters, layers_activations)
            
            # Compute cost.
            cost = self.compute_cost(AL, Y, self.parameters,lambd)
            
            # Backward propagation.
            grads = self.backward_propagation(AL, Y, caches, lambd, layers_activations)
            
            # Update parameters.
            self.parameters = self.update_parameters(self.parameters, grads, learning_rate)
                
            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print(f"Cost after iteration {i + 1}: {np.squeeze(cost)}")

  
    def predict(self, X_test, layers_activations):
        Y_hat, _ = self.forward_propagation(X_test, self.parameters, layers_activations)
        Y_pred = np.where(Y_hat >= 0.5, 1, 0).T

        return Y_pred