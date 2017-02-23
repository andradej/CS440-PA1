"""
LogisticRegression.py

CS440/640: Lab-Week5

Lab goal: 1) Implement logistic regression classifier
"""

import numpy as np 
import matplotlib.pyplot as plt 
import csv
from sklearn.metrics import confusion_matrix
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
        self.input_dim = input_dim
        self.output_dim = output_dim
        
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
        for i in range(len(X)):
            one_hot_y = np.zeros(self.output_dim)
            one_hot_y[int(y[i])] = 1
            
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
        z = np.dot(X, self.theta) + self.bias
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
                        
            #forward propagation
            z = np.dot(X, self.theta) + self.bias
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                
            differences = []

            #generating difference matrix
            for i in range(len(X)):
                one_hot_y = np.zeros(self.output_dim)
                one_hot_y[int(y[i])] = 1
                    
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


################################################################################    
linear = True
if linear:
    X_values = np.genfromtxt('DATA/Linear/X.csv', delimiter=",")
    y_values = np.genfromtxt('DATA/Linear/y.csv', delimiter=",")
else:
    X_values = np.genfromtxt('DATA/NonLinear/X.csv', delimiter=",")
    y_values = np.genfromtxt('DATA/NonLinear/y.csv', delimiter=",")

#print(y_values[0])

v = LogisticRegression(2,2)

# print(v.compute_cost(X_values, y_values))
v.fit(X_values, y_values)
# # print(v.predict(X_values))
plot_decision_boundary(v, X_values, y_values)

################################################################################    
# Question 6
dig = LogisticRegression(64,10)
X_dig = np.genfromtxt('DATA/Digits/X_train.csv', delimiter=",")
y_dig = np.genfromtxt('DATA/Digits/y_train.csv', delimiter=",")

dig.fit(X_dig, y_dig)

X = np.genfromtxt('DATA/Digits/X_test.csv', delimiter=',')
y_pred = dig.predict(X)
y_actual = np.genfromtxt('DATA/Digits/y_test.csv', delimiter=',')

m = confusion_matrix(y_actual, y_pred)
print(str(m))

################################################################################    
predictions = dig.predict(X_dig)
correct = 0

for i in range(len(predictions)):
    if predictions[i] == y_dig[i]: # Check if it was predicted correctly
        correct += 1

correct /= len(predictions)
print("Accuracy: " + str(correct * 100) + "%")


