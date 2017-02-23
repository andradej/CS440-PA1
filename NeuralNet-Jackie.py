"""
LogisticRegression.py

CS440/640: Lab-Week5

Lab goal: 1) Implement logistic regression classifier
"""

import numpy as np 
import matplotlib.pyplot as plt 
import csv
from sklearn.metrics import confusion_matrix

np.seterr(divide='ignore', invalid='ignore', over='ignore') # https://docs.scipy.org/doc/numpy/reference/generated/numpy.seterr.html
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
        self.theta_hidden = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
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
        activation = sigmoid(z)
        z_hidden = np.dot(activation, self.theta_hidden) + self.bias_hidden
        exp_z = np.exp(z_hidden)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        # Calculate the cross-entropy loss
        cross_ent_err = -np.log(softmax_scores[range(len(X)), y.astype('int64')])
        data_loss = np.sum(cross_ent_err)
        return 1./len(X) * data_loss
#
#        return (1./len(X)) * data_loss
        
        #calculate the cost of each score
#        calc_loss = []
#        for i in range(len(X)):
#            one_hot_y = np.zeros(len(X[0]))
#            one_hot_y[int(y[i])] = 1
#            
##            print(one_hot_y)
##            print(softmax_scores[i])
#            calc_loss.append(-np.sum(one_hot_y * np.log(softmax_scores[i])))
#        
#        total_loss = sum(calc_loss)
#        
#        return 1./len(calc_loss) * total_loss



    
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
        activation = sigmoid(z)
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
            
        for i in range(0,5000):
            # Do Forward propagation to calculate our predictions
            z1 = X.dot(self.theta) + self.bias
            a1 = sigmoid(z1)
            z2 = a1.dot(self.theta_hidden) + self.bias_hidden
            exp_z = np.exp(z2)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            # Back Propagation
            delta3 = softmax_scores
            delta3[range(len(X)), y.astype('int64')] -= 1
            dw2 = (a1.T).dot(delta3) 
            db2 = np.sum(delta3, axis = 0, keepdims = True)
            delta2 = delta3.dot(self.theta_hidden.T) * sigmoidPrime(z1)
            dw1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis = 0)
            
            # Gradient descent parameter uqdate
            self.theta -= self.epsilon * dw1
            self.bias -= self.epsilon * db1
            self.theta_hidden -= self.epsilon * dw2
            self.bias_hidden -= self.epsilon * db2
#        for i in range(1000):
#
#            # forward propogation with hidden layer and tanh activation function
#            z = np.dot(X, self.theta) + self.bias
#            activation = np.tanh(z)
#            z_hidden = np.dot(activation, self.theta_hidden) + self.bias_hidden
#            exp_z = np.exp(z_hidden)
#            #contains probabilities of either 0 or 1 occuring
#            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
#            
#            #backward propogation
#            delta3 = softmax_scores
#            for i in range(len(X)):
#                delta3[i][y.astype(int)] -= 1
#
#
#            dW2 = np.dot(np.transpose(activation), delta3)
#            db2 = np.sum(delta3, axis=0, keepdims=True)
#            delta2 = np.dot(delta3, np.transpose(self.theta_hidden)) * (1 - np.power(activation,2))
#            dW1 = np.dot(np.transpose(X), delta2)
#            db1 = np.sum(delta2, axis=0)
#
#            
#            self.theta -= self.epsilon * dW1
#            self.bias -= self.epsilon * db1
#            self.theta_hidden -= self.epsilon * dW2
#            self.bias_hidden -= self.epsilon * db2
                    
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

def sigmoid(z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
def sigmoidPrime(z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
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

v = NeuralNet(2,2,10,0.01)
#print(v.compute_cost(X_values, y_values))

# print(v.predict(X_values))
v.fit(X_values, y_values)
plot_decision_boundary(v, X_values, y_values)


# Question 7
dig = NeuralNet(64, 10, 10, 0.01)
X_dig = np.genfromtxt('DATA/Digits/X_train.csv', delimiter=",")
y_dig = np.genfromtxt('DATA/Digits/y_train.csv', delimiter=",")

dig.fit(X_dig, y_dig)

X = np.genfromtxt('DATA/Digits/X_test.csv', delimiter=',')
y_pred = dig.predict(X)
y_actual = np.genfromtxt('DATA/Digits/y_test.csv', delimiter=',')

m = confusion_matrix(y_actual, y_pred)
print(str(m))

predictions = dig.predict(X_dig)
correct = 0

for i in range(len(predictions)):
    if predictions[i] == y_dig[i]: # Check if it was predicted correctly
        correct += 1

correct /= len(predictions)
print("Accuracy: " + str(correct * 100) + "%")
            
    