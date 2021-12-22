# SVM
Support Vector Machine for Classification from scrath
####
SVM.py file usage guide

1. Import SVM into the script you are working on.
Ex. import SVM as SVM
2. To import the data use the Load_data () method entering the file path.
Note: The file must be ordered so that the last column corresponds to the labels.
3. Separate the training and test sets using the Split_data () method entering the previously loaded dataset and optionally the random_state seed.
4. Run the SVM method which has the following parameters by default.
SVM (X_train, y_train, n_iter = 100, epsilon = 1e-4, lr = 1e-3, l = 1, Debuging = True, batch_size = 1, Adam = False, B1 = 0.9, B2 = 0.999, random_state = None) :
n_iter - number of interactions
epsilon - Delta J at which the method stops.
lr - Learning rate.
l - Regularization factor.
Debugging - True to graph the cost during training.
batch_size - Size of the batch.
Adam - True for Adam optimizer, otherwise gradient descent is applied.
B1 and B2 - Adam optimizer parameters.
random_state - Random seed for initialization of weights.
 
*** This method returns the weight matrix w and the bias b.

5. Make predictions on the test data using the Predict () method by entering the X_test, weights w, and bias b.
6. Obtain the performance metrics Accuracy and confusion matrix using the Confussion_Matriz method by entering the predicted data and the actual labels. This method returns the confusion matrix and the accuracy.
