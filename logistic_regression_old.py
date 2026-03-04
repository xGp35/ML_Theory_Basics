import math

#Sigmoid Function
def sigmoid(z):
    return 1/ (1+math.exp(-z))

# Predict probability for single example
def predict_proba(x, w, b):
    # x and w are lists of same length
    z = sum(w_i*x_i for w_i, x_i in zip(w,x)) + b
    return sigmoid(z)

# Define Binary Cross Entropy Loss for one example
def BinaryCrossEntropy(y, p):
    eps = 1e-15 #avoid log(0)
    p = max(min(p, 1-eps), eps)
    return -(y*math.log(p) + (1-y)*math.log(1-p))

def binary_cross_entropy_batch(y_true, y_pred):
    """
    Docstring for binary_cross_entropy_batch
    
    :param y_true: list of 0's and 1's
    :param y_pred: list of predicted probabilities
    """
    total_loss = 0
    n = len(y_true)
    eps = 1e-15
    
    for y, p in zip(y_true, y_pred):
        p = max(min(p,1-eps), eps)
        total_loss += -(y*math.log(p) + (1-y)*math.log(1-p))
    return total_loss/n
    

# Example usage
x = [1.0, 2.0]
w = [0.5, -0.25]
b = 0.1
y = 1

p = predict_proba(x,w,b)
loss = BinaryCrossEntropy(y,p)

print("Predicted Probability: ",p)
print("loss: ", loss)


y_true = [1, 0, 1, 1]
y_pred = [0.9, 0.2, 0.7, 0.6]

loss = binary_cross_entropy_batch(y_true, y_pred)
print("Average loss:", loss)