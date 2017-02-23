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
            hidden_dim: Number of nodes in the hidden layer
            epsilon: learning rate
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

        # Calculate the cost per point and then return the average cost
        cost_per_point = -np.log(softmax_scores[range(len(X)), y.astype('int64')])
        total_loss = np.sum(cost_per_point)
        return 1./len(X) * total_loss
    
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

################################################################################    
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

################################################################################    


predictions = dig.predict(X_dig)
correct = 0

for i in range(len(predictions)):
    if predictions[i] == y_dig[i]: # Check if it was predicted correctly
        correct += 1

correct /= len(predictions)
print("Accuracy: " + str(correct * 100) + "%")
################################################################################    

# Question 4
def learning_rate_graph(X, y, rate):
    graph_y = [0] * 5 #number of trials
    for i in range(len(graph_y)):
        NN = NeuralNet(2,2,5,rate)
        NN.fit(X, y)
        graph_y[i] = NN.compute_cost(X,y)
        
    return graph_y

# Question 5
def hidden_nodes_graph(X, y, num_nodes):
    graph_y = [0] * 10 #number of trials
    for i in range(len(graph_y)):
        NN = NeuralNet(2,2,num_nodes,0.01)
        NN.fit(X, y)
        graph_y[i] = NN.compute_cost(X,y)
        
    return graph_y

# The code below is to graph costs 
x = [x * 1 for x in range(10)]
#rates = [0.01, 0.1, 0.2]
#graph_y_first = learning_rate_graph(X_values, y_values, rates[0])
#graph_y_second = learning_rate_graph(X_values, y_values, rates[1])
#graph_y_third = learning_rate_graph(X_values, y_values, rates[2])
#plt.plot(x,graph_y_first,'r--',x,graph_y_second, '--b',x,graph_y_third, '--g')
#plt.title("different learning rate")
#plt.show()

num_nodes = [5, 10, 15]
graph_y_first = hidden_nodes_graph(X_values, y_values, num_nodes[0])
graph_y_second = hidden_nodes_graph(X_values, y_values, num_nodes[1])
graph_y_third = hidden_nodes_graph(X_values, y_values, num_nodes[2])
plt.plot(x, graph_y_first, 'r--',x,graph_y_second, '--b',x,graph_y_third, '--g')
plt.title("different number of nodes")
plt.show()         
    