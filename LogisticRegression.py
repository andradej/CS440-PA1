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
            
        z = np.dot(X, self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        #print("SOOOFFTTTTOOOO MAAAXXEEEUUUUU DESNU-KA: " + str(softmax_scores))
        #print(len(softmax_scores))
        
        mean_cost = 0
        errors = 0
        for i in range(len(X)):
            if int(y[i]) == 0:
                one_hot_y = np.array([1,0])
            elif int(y[i]) == 1:
                one_hot_y = np.array([0,1])
            else:
                errors += 1
            
            cost_for_sample = -np.sum(one_hot_y * np.log(softmax_scores[i]))
            
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
            
        current_cost = self.compute_cost(X,y)
        prev_cost = 0.0
        difference_of_costs = current_cost - prev_cost
        convergence_point = 0.0001 #want to be close to 0 since can't actually reach true 0
        
        #we loop until costs we are computing are changing minimally
        # ^ aka we have reached a "convergence"
        while difference_of_costs >= convergence_point:
            
            #current_cost = self.compute_cost(X,y)
            
#            print("Welcome to a new iteration!")
#            print("current_cost: " + str(current_cost))
#            print("prev_cost: " + str(prev_cost))
                        
            #forward propagation
            z = np.dot(X, self.theta) + self.bias
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                
            differences = []
            #generating difference matrix
            for i in range(len(X)):
                #backward propagation:
                if int(y[i]) == 0:
                    one_hot_y = np.array([1,0])
                elif int(y[i]) == 1:
                    one_hot_y = np.array([0,1])
                    
                difference = softmax_scores[i] - one_hot_y
                
                differences.append(difference)
            
            gradient_wrt_weight = np.dot(np.transpose(X), differences)
            gradient_wrt_bias = np.dot(np.transpose(np.ones((len(X), 1))), differences)
            
            #w = w - learning_rate * gradient of cost w.r.t weights
            self.theta = self.theta - 0.001 * gradient_wrt_weight
            
            #b = b - learning_rate * gradient of cost w.r.t. biases
            self.bias = self.bias - 0.001 * gradient_wrt_bias
            
            prev_cost = current_cost
            current_cost = self.compute_cost(X,y)
            difference_of_costs = abs(current_cost - prev_cost)
            
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

X_values = np.genfromtxt('DATA/Linear/X.csv', delimiter=",")
y_values = np.genfromtxt('DATA/Linear/y.csv', delimiter=",")
#print(y_values[0])

v = LogisticRegression(2,2)
#print(v.compute_cost(X_values, y_values))
print(v.fit(X_values, y_values))
#print(v.predict(X_values))
plot_decision_boundary(v, X_values, y_values)
            
    