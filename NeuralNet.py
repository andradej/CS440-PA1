"""
LogisticRegression.py

CS440/640: Lab-Week5

Lab goal: 1) Implement logistic regression classifier
"""

import numpy as np 
import matplotlib.pyplot as plt 
import csv


class NeuralNet:
    """
    This class implements a Neural Net
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim, epsilon):
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """
        
        self.theta = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.bias = np.zeros((1, hidden_dim))
        
        # initialize a hidden layer 
        self.theta_hidden = np.random.randn(hidden_dim, output_dim) / np.sqrt(input_dim)
        self.bias_hidden = np.zeros((1, output_dim))
        
        #used in gradient descent
        self.epsilon = epsilon
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total cost on the dataset.
        
        args:
            X: Data array
            y: Labels corresponding to input data
        
        returns:
            cost: average loss per data sample
        """
        #TODO:
    
        z = np.dot(X, self.theta) + self.bias
        activation = np.tanh(z)
        z_hidden = np.dot(activation, self.theta_hidden) + self.bias_hidden
        exp_z = np.exp(z_hidden)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        #calculate the cost of each score
        calc_loss = []
        for i in range(len(X)):
            if int(y[i]) == 0:
                one_hot_y = np.array([1,0])
            elif int(y[i]) == 1:
                one_hot_y = np.array([0,1])
            else:
                errors += 1
        calc_loss.append(-np.sum(one_hot_y * np.log(softmax_scores[i])))
        
        total_loss = sum(calc_loss)
        return 1./len(calc_loss) * total_loss
        

    
    #--------------------------------------------------------------------------
 
    def predict(self,X):
        """
        Makes a prediction based on current model parameters.
        
        args:
            X: Data array
            
        returns:
            predictions: array of predicted labels
        """
        # forward propogation with hidden layer and tanh activation function
        z = np.dot(X, self.theta) + self.bias
        activation = np.tanh(z)
        z_hidden = np.dot(activation, self.theta_hidden) + self.bias_hidden
        exp_z = np.exp(z_hidden)
        #contains probabilities of either 0 or 1 occuring
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        #chooses 0 or 1 based upon the highest probabiity 
        predictions = np.argmax(softmax_scores, axis = 1)
        return predictions
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y):
        """
        Learns model parameters to fit the data.
        """  
        #TODO: 
            
        current_cost = self.compute_cost(X,y)
        prev_cost = 0.0
        difference_of_costs = current_cost - prev_cost
        convergence_point = 0.0001 #want to be close to 0 since can't actually reach true 0
        
        #we loop until costs we are computing are changing minimally
        # ^ aka we have reached a "convergence"
        for i in range(1000):

            # forward propogation with hidden layer and tanh activation function
            z = np.dot(X, self.theta) + self.bias
            activation = np.tanh(z)
            z_hidden = np.dot(activation, self.theta_hidden) + self.bias_hidden
            exp_z = np.exp(z_hidden)
            #contains probabilities of either 0 or 1 occuring
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            #backward propogation
            delta3 = softmax_scores
            delta3[range(len(X)), y] -= 1
            dW2 = np.dot(np.transpose(activation), delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = np.dot(delta3, np.transpose(self.theta_hidden)) * (1 - np.power(activation,2))
            dW1 = np.dot(np.transpose(X), delta2)
            db1 = np.sum(delta2, axis=0)
            
            self.theta += -self.epsilon * dW1
            self.bias += -self.epsilon * db1
            self.theta_bias += -self.epsilon * dW2
            self.bias_hidden += -self.epsilon * db2
                    
#            beta_error = []
#            beta_other_nodes = []
#            #generating difference matrix
#            for i in range(len(X)):
#                #backward propagation:
#                if int(y[i]) == 0:
#                    one_hot_y = np.array([1,0])
#                elif int(y[i]) == 1:
#                    one_hot_y = np.array([0,1])
#                
#                # calculate the error: beta = desired - output
#                beta_z = one_hot_y - softmax_scores[i]
#                
#                # calculate for all other nodes: beta_j = output * (1- output)*beta_z
#                beta_j = self.epsilon * softmax_scores[i] * (1-softmax_scores[i]) * beta_z
#                    
#                beta_error.append(beta_z)
#                beta_other_nodes.append(beta_j)
                #differences.append(beta_z)
                
#            for i in range(len(softmax_scores)):
#                beta_h = np.zeros((2, 1))
#                temp = softmax_scores[i] * (1 - softmax_scores[i]) * beta_error[i]
#                temp.shape(2, 1)
#                
#                beta_h += np.dot(self.theta, temp)
#                beta_h.shape = (3, )
#                
#                x = X[i]
#                x.shape = (x.shape[0], 1)
#                h_delta += self.epsilon * x * (beta_j[i]*(1-beta_j[i])*beta_h)
#                
#            
            #print("beta error: " + str(beta_error))
            #print("beta_other_nodes: " + str(beta_other_nodes))
            
#            self.theta += np.dot(np.transpose(softmax_scores), beta_other_nodes)
#            self.theta_hidden += h_delta
            #print(self.theta)
            
#            gradient_wrt_weight = np.dot(np.transpose(X), differences)
#            gradient_wrt_bias = np.dot(np.transpose(np.ones((len(X), 1))), differences)
#            
#            self.theta_hidden = self.theta_hidden - self.epsilon * gradient_wrt_weight
#            
#            #b = b - learning_rate * gradient of cost w.r.t. biases
#            self.bias = self.bias - self.epsilon * gradient_wrt_bias
#            self.bias_hidden = self.bias_hidden - self.epsilon * gradient_wrt_bias
#            
#            prev_cost = current_cost
#            current_cost = self.compute_cost(X,y)
#            difference_of_costs = abs(current_cost - prev_cost)
            
        return self

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def plot_decision_boundary(model, X, y):
    """
    Function to print the decision boundary given by model.
    
    args:
        model: model, whose parameters are used to plot the decision boundary.
        X: input data
        y: input labels
    """
    
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()

# helper function sigmoid to determine the cost
def sigmoid(z):
    return 1 / (1 + (np.e ** (-1 * z)))

################################################################################    

linear = True
if linear:
    X_values = np.genfromtxt('DATA/Linear/X.csv', delimiter=",")
    y_values = np.genfromtxt('DATA/Linear/y.csv', delimiter=",")
else:
    X_values = np.genfromtxt('DATA/NonLinear/X.csv', delimiter=",")
    y_values = np.genfromtxt('DATA/NonLinear/y.csv', delimiter=",")


#print(y_values[0])

v = NeuralNet(2,2,2,0.01)
print(v.compute_cost(X_values, y_values))
print(v.fit(X_values, y_values))
print(v.predict(X_values))
plot_decision_boundary(v, X_values, y_values)
            
    