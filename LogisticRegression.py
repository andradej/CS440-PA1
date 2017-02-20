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
        
        total_cost = 0
        errors = 0
        for i in range(len(X)):
            if int(y[i]) == 0:
                one_hot_y = np.array([1,0])
            elif int(y[i]) == 1:
                one_hot_y = np.array([0,1])
            else:
                errors += 1
            
            cost_for_sample = -np.sum(one_hot_y * np.log(softmax_scores))
            
            total_cost += cost_for_sample
            
        total_cost /= len(X)
            
            
        return total_cost

    
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
            
        for i in range(len(X)):
            #forward propagation
            z = np.dot(X,self.theta) + self.bias
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            #backward propagation compute the gradient of the cost w.r.t. your weights/biases and update them
            
            
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


################################################################################    

X_values = np.genfromtxt('DATA/Linear/X.csv', delimiter=",")
y_values = np.genfromtxt('DATA/Linear/y.csv', delimiter=",")
#print(y_values[0])

v = LogisticRegression(1000,1)
print(v.compute_cost(X_values, y_values))
#print(v.predict(X_values))
            
    