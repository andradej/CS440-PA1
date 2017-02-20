"""
LogisticRegression.py

CS440/640: Lab-Week5

Lab goal: 1) Implement logistic regression classifier
"""

import numpy as np 
import matplotlib.pyplot as plt 
import csv
'''
with open('DATA/Linear/y.csv', 'r') as f:
    reader = csv.reader(f, delimiter=",")
    y_values = []
    for row in reader:
        y_values.append(int(row[0][0]))
        
    print(y_values)
    
print(y_values)
'''

class LogisticRegression:
    """
    This class implements a Logistic Regression Classifier.
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """
        
        self.theta = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.bias = np.zeros((1, output_dim))
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total cost on the dataset.
        
        args:
            X: Data array
            y: Labels corresponding to input data
        
        returns:
            cost: average cost per data sample
        """
        #TODO:
            
        z = np.dot(np.transpose(self.theta), X) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        mean_cost = 0
        errors = 0
        for i in range(len(X)):
            if int(y[i]) == 0:
                one_hot_y = np.array([1,0])
            elif int(y[i]) == 1:
                one_hot_y = np.array([0,1])
            else:
                errors += 1
            
            cost_for_sample = -np.sum(one_hot_y * np.log(softmax_scores))
            
            mean_cost += cost_for_sample
            
        mean_cost /= len(X)
            
            
        return mean_cost

    
    #--------------------------------------------------------------------------
 
    def predict(self,X):
        """
        Makes a prediction based on current model parameters.
        
        args:
            X: Data array
            
        returns:
            predictions: array of predicted labels
        """
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)
        return predictions
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y):
        """
        Learns model parameters to fit the data.
        """  
        #TODO:
            
        learning_rate = 0.1
        w = 0
        b = 0
            
        print(X)
        for i in range(len(X)):
            #forward propagation
            z = np.dot(np.transpose(self.theta), X) + self.bias
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            #backward propagation: compute the gradient of the cost w.r.t. your weights/biases and update them
            #dot product of input X with the difference between your predictions (softmax_scores) and the ground truth (one_hot_y)
            if int(y[i]) == 0:
                one_hot_y = np.array([1,0])
            elif int(y[i]) == 1:
                one_hot_y = np.array([0,1])
                
            difference = softmax_scores - one_hot_y
            
            weight_cost = (y[i] / sigmoid(z)) + ((1-y[i]) / (1-sigmoid(z)))
            print("weight_cost: " + str(weight_cost))
            
            gradient_wrt_weight = np.dot(X, np.transpose(difference))
            
            gradient_wrt_bias = np.dot(np.ones((len(X), 1)), difference)
            
            # cost (derivative of cost formula in compute_cost)
            weight_cost = (y[i] / sigmoid(z)) + ((1-y[i]) / (1-sigmoid(z)))
            
            w = w - learning_rate * gradient_wrt_weight
            b = b - learning_rate * gradient_wrt_bias
            
            #print("difference: " + str(difference))
            #print(gradient_wrt_weight)
            #print(gradient_wrt_bias)
            break
            
            
            
        return 0

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

X_values = np.genfromtxt('DATA/Linear/X.csv', delimiter=",")
y_values = np.genfromtxt('DATA/Linear/y.csv', delimiter=",")
#print(y_values[0])

v = LogisticRegression(1000,1)
print(v.compute_cost(X_values, y_values))
print(v.fit(X_values, y_values))
#print(v.predict(X_values))
            
    