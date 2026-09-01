<H3>ENTER YOUR NAME: sai kripa sk </H3>
<H3>ENTER YOUR REGISTER NO.: 212224040284</H3>
<H3>EX. NO.5</H3>
<H3>DATE: 01-09-26</H3>
<H1 ALIGN =CENTER>Implementation of XOR  using RBF</H1>
<H3>Aim:</H3>
To implement a XOR gate classification using Radial Basis Function  Neural Network.

<H3>Theory:</H3>
<P>Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows XOR truth table </P>

<P>XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below </P>




<P>The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.
A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.
A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.
</P>





<H3>ALGORITHM:</H3>
Step 1: Initialize the input  vector for you bit binary data<Br>
Step 2: Initialize the centers for two hidden neurons in hidden layer<Br>
Step 3: Define the non- linear function for the hidden neurons using Gaussian RBF<br>
Step 4: Initialize the weights for the hidden neuron <br>
Step 5 : Determine the output  function as 
                 Y=W1*φ1 +W1 *φ2 <br>
Step 6: Test the network for accuracy<br>
Step 7: Plot the Input space and Hidden space of RBF NN for XOR classification.

## PROGRAM:
```py
!pip install numpy
!pip install matplotlib

import numpy as np
import matplotlib.pyplot as plt

# Gaussian Radial Basis Function
def gaussian_rbf(x, landmark, gamma=1):
    return np.exp(-gamma * np.linalg.norm(x - landmark) ** 2)
def end_to_end(X1, X2, ys, mu1, mu2):
    # Transform input using RBF
    from_1 = []
    from_2 = []

    for x1, x2 in zip(X1, X2):
        point = np.array([x1, x2])
        from_1.append(gaussian_rbf(point, mu1))
        from_2.append(gaussian_rbf(point, mu2))

    # Plot
    plt.figure(figsize=(13, 5))

    plt.subplot(1, 2, 1)
    plt.scatter((X1[0], X1[3]), (X2[0], X2[3]), label="Class_0")
    plt.scatter((X1[1], X1[2]), (X2[1], X2[2]), label="Class_1")
    plt.xlabel("$X1$", fontsize=15)
    plt.ylabel("$X2$", fontsize=15)
    plt.title("XOR: Linearly Inseparable", fontsize=15)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(from_1[0], from_2[0], label="Class_0")
    plt.scatter(from_1[1], from_2[1], label="Class_1")
    plt.scatter(from_1[2], from_2[2], label="Class_1")
    plt.scatter(from_1[3], from_2[3], label="Class_0")

    plt.plot([0, 0.95], [0.95, 0], "k--")
    plt.annotate(
        "Separating hyperplane",
        xy=(0.4, 0.55),
        xytext=(0.55, 0.66),
        arrowprops=dict(facecolor="black", shrink=0.05)
    )

    plt.xlabel(f"$mu1$: {mu1}", fontsize=15)
    plt.ylabel(f"$mu2$: {mu2}", fontsize=15)
    plt.title("Transformed Inputs: Linearly Separable", fontsize=15)
    plt.legend()
    plt.show()

    # Create matrix A
    # AW = Y
    A = []

    for i, j in zip(from_1, from_2):
        temp = []
        temp.append(i)
        temp.append(j)
        temp.append(1)       # Bias
        A.append(temp)

    A = np.array(A)
    # Calculate weights using pseudo-inverse
    W = np.linalg.pinv(A).dot(ys)
    print(f"Weights: {W}")
    return W


def predict_matrix(point, weights):
    # type your code
    # Transform input using the same RBF centers
    rbf1 = gaussian_rbf(point, mu1)
    rbf2 = gaussian_rbf(point, mu2)

    # Add bias
    A = np.array([rbf1, rbf2, 1])

    # Prediction
    prediction = A.dot(weights)
    return np.round(prediction)

# XOR input
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])

# Target output
ys = np.array([0, 1, 1, 0])

# Centers
mu1 = np.array([0, 1])
mu2 = np.array([1, 0])

# Train the RBF model
w = end_to_end(x1, x2, ys, mu1, mu2)

# Testing
print(f"Input: {np.array([0, 0])}, Predicted: {int(predict_matrix(np.array([0, 0]), w))}")
print(f"Input: {np.array([0, 1])}, Predicted: {int(predict_matrix(np.array([0, 1]), w))}")
print(f"Input: {np.array([1, 0])}, Predicted: {int(predict_matrix(np.array([1, 0]), w))}")
print(f"Input: {np.array([1, 1])}, Predicted: {int(predict_matrix(np.array([1, 1]), w))}")
```
<H3>OUTPUT:</H3>

<img width="1486" height="647" alt="image" src="https://github.com/user-attachments/assets/71fd0e61-2abd-4240-8929-bb5e536be893" />

<H3>Result:</H3>
Thus , a Radial Basis Function Neural Network is implemented to classify XOR data.








