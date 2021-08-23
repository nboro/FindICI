# Hyper-parameters Configurations

For hyper-parameter tuning we used **grid search**, an exhaustive search algorithm through a manually-specified subset of the parameter space.
Please consider that we could not employ k-fold cross-validation and grid search for CNNs and LSTMs. 
Indeed, exhaustive hyper-parameter searching for these two deep learning models is dramatically slow as previously observed. 
Instead, following comparable previous studies, we manually tested different hyper-parameter configurations for each model and selected the best-performing configuration.
We experimented with the following designs and hyper-parameters for the classifiers.


## Random Forest

We tuned the parameters for the RF classifier motivated by the previous work~\cite{DeepLiguistic}. Specifically, the number of trees varied between 100 and 1000, the maximum depth of the tree varied between 80 and 110, the minimum number of samples required to split an internal node was between 8 and 12, the minimum number of samples required to be at a leaf node was between 3 and 5. Finally, the samples were bootstrapped when building the trees.

|Param| Values|
|---|----|
|bootstrap|True|
|max_depth|[80, 90, 100, 110]|
|max_features|[2, 3]|
|min_samples_leaf|[3, 4, 5]|
|min_samples_split|[8, 10, 12]|
|n_estimators|[100, 200, 300, 1000]|


## Support Vector Machine

We used the Radial Basis Function kernel, which has demonstrated good performance in previous works. 
Furthermore, we tuned the parameters C and Gamma by searching the optimal values between 0.1 and 1000 for C and between 0.0001 and 1 for gamma.

|Param| Values|
|---|----|
|C|[0.1, 1, 10, 100, 1000]|
|gamma|[1, 0.1, 0.01, 0.001, 0.0001]|

## eXtreme Gradient Boosting

We tuned the learning rate parameters with values ranging between 0.05 and 0.30, the maximum depth of the generated trees from 10 to 50, and the minimum sum of instance hessian weight needed in a child node between 1 to 7. 
The number of generated trees was set to 100.

|Param| Values|
|---|----|
|learning_rate|[0.05, 0.10, 0.30]|
|max_depth|[10, 30, 50]|


## Multi-Layer Perceptron

For the parameters of the MLP, we used Rectified Linear Unit (ReLU) as the activation function. 
We applied the L2 penalty by setting the alpha parameter to 0.001 to prevent overfitting. 
The maximum number of iterations was set to 50, while the learning rate parameter to be adaptive to keep the learning rate constant as long as training loss keeps decreasing. 
We searched for the optimal number of hidden layers and neurons used in the hidden layers for values between 100 and 600. 
Finally, we added to the search parameters two optimizers, namely Stochastic Gradient Descent (SGD) and the first-order gradient-based optimization of stochastic objective functions (Adam).

|Param| Values|
|---|----|
|hidden_layer_sizes|[(randint.rvs(100,600,1), randint.rvs(100,600,1),), (randint.rvs(100,600,1),)]|
|solver|['sgd', 'adam']|



## Convolutional Neural Networks 

We use two convolutional pooling layers to reduce data dimensionality and capture the local features, similarly to the previous work. 
We used L2 regularization in each convolutional layer and a dropout layer (for the 25% of the inputs) to prevent overfitting. 
A dense layer combines the previously captured local features by the convolutional and subsampling layers. 
The dense layer's output vector predicts and detects inconsistent module use within a task. 
The output layer consists of neurons per one task of each module. 
The output neurons are 0 (inconsistent) or 1 (consistent). 
The measure for the loss function is the _Mean Absolute Error_ (MAE), and the corresponding optimizer is the _Stochastic Gradient Descent_ (SGD).

## Long Short-Term Memory 
The model is trained using the Adam optimizer. 
Besides a regular dropout layer and motivated from the prior work, we perform recurrent dropout to mask the connections between the recurrent units in order to prevent overfitting. 
We use binary cross-entropy as the loss function and sigmoid activation function at the last dense layer to predict the label. 
The number of training epochs was set to 20, and the batch size of the input sequences was 30. 
The best-performing parameters were defined after running the network multiple times using different parameter setups.
