# **CONVOLUTIONAL NEURAL NETWORK IMAGE CLASSIFIER :**

This readme summarizes the work completed for a Fashion Brand Predictor Neural Network.


# **ABSTRACT:**

The goal of the project is to identify the brand name and price of a given clothing image using deep learning process. 


# **DATA PREPARATION:**
The dataset we are working with is the Farfetch Listings obtained from kaggale website.

1. Data cleaning:
*   Check for missing data and place holders
*   Check for data types
*   Extract a sub-dataset to work on the project of interest 
*   Separate independant and target variables as X and Y
*   Download URL images and import it as a numpy array using OpenCV (Open Source Computer Vision Library) 


2. Check data distribution for regression problems

![alt text](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(1).png)

* The price/target variable for regression analysis is not normally distributed, so log transfer was applied to normally distribute the price column, at the end the process will be reversed to predict the real price. 

![alt text](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(2).png)

3. check data balance for classification problems

![Brand Distribution for a multi-classifier model](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download.png)

*   **Brand Distribution/target labels sounds balanced for a multi-classifier**  

4. Solving class imbalancement problem for binary classification :
*   Generating labels to help the balance

 ![before generating labels](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(3).png)

  ![after generating labels](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(8).png)


* Generating image for added labels through image augmentation

  ![original image](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(4).png)
  
  ![fliped image](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(5).png)

  ![rotated image](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(6).png)

  ![rotated image](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(7).png)


5. splitting train and test data
6. shuffling data for more accurate predictions 

 [Data_preprocessing_notebook](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/notebooks/Data_Preprocessing.ipynb)
 
 
 
 # **MODELING APPROACHES**:

this project approached the problem from six different ways:
1. **Multi-class neural network classifier**

* For predicting brand names 

 [multiclass_Classifier1_notebook](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/notebooks/multiclass_classifier1.ipynb)

 * Model Accuracy, Relu and Softmax activation functions

 ![Relu activation](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(14).png)

 [multiclass_Classifier2_notebook](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/notebooks/multiclass_classifier2.ipynb)

  * Model Accuracy, Sigmoid and Softmax activation functions, decreasing drop out and batch normalization layers in the model 

 ![sigmoid activation](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(15).png)

 [multiclass_Classifier3_notebook](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/notebooks/multiclass_classifier3.ipynb) 

   * Model Accuracy, tanh and Softmax activation functions, dropping more hidden layers from the model

 ![tanh activation](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(17).png)

As we can see, the accuracy of the models is not satisfying.

3. **Transfer learning for multi-classifiers**:
* Transfer learning uses pre-trained models when developing a new convolutional neural network.
* One or more layers and their weights from the previously trained model are used as a starting point for the training process of the new model on new but related problems.

 [transfer_learning1_notebook](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/notebooks/Transfer_learning1.ipynb) 

 ![alt text](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(18).png)

 [transfer_learning2_notebook](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/notebooks/Transfer_learning2.ipynb)

 * Adding more layers to the custom model to higher the accuracy, but as shown below, it led to an overfit model

 ![alt text](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(19).png)

 [transfer_learning3_notebook](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/notebooks/Transfer_learning3.ipynb)
 * Dropping drop out layers, adding more convolutional layers to the custom model to reduce the number of trainable parameters in fully connected layers, because a model with too much capacity can learn it too well and it causes the overfitting and fially, adding bias_regularizer to the fully connected layers in order to reduce overfitting and increase the accuracy.

 ![alt text](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(20).png)

* One more try:
 
 ![alt text](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(21).png)

* our model still overfits the training data, also the accuracy of the validation is not acceptable, so we are moving to our next approach to compare the results. 

4. **One-vs-all approach**: 
*   This approach splits the multi-class classification dataset into multiple binary classification datasets and fit a binary classification model on each.

*   Each model predicts a class membership probability or a probability-like score. The argmax of these scores (class index with the largest score) is then used to predict a class.

*   This approach is fairly reasonable when the total number of classes is small, but becomes inefficient as the number of classes rises.

  [one_vs_all_notebook](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/notebooks/one_vs_all.ipynb)

 ![alt text](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(22).png)
*  the model accuracy was not satisfying, part of that was related to the imbalanced count of data for one brand vs the rest of brands in a binary model, so in the next approach we try to balance data for each binary model and use transfer learning to higher the accuracy of each binary model. 

4. **Combination of One-vs-all and Transfer learning**

  [final_capstone_notebook](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/one_vs_all+transfer_learning.ipynb)

* Model accuracy for Chanel brand binary prediction

  ![Channel](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(9).png) 

* Model accuracy for Dolce & Gabbana brand binary prediction

  ![Dolce & Gabbana](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(10).png)
* Model accuracy for Prada brand binary prediction

  ![Prada](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(11).png)

* Model accuracy for Saint Laurent brand binary prediction

  ![Saint Laurent](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(12).png)
* Model accuracy for Gucci brand binary prediction

  ![Gucci](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/download%20(13).png)



5. **Neural network regression model**
*   For predicting price 

  [regression_classifier_notebook](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/notebooks/Regression_classifier.ipynb)


6. **Combination of regression and classification models**
*   For predicting brand mane and price


# **CONCLUSION**
After exploring many different models and approaches and comparing the results, the best result goes to the **Combination of One-vs-all and Transfer learning**.

![alt text](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/(1307).png)

The model shows %66 success rate at predicting the brand name.
![alt text](https://github.com/FarnazG/dsc-capstone-project-v2-online-ds-ft-120919/blob/master/images/Screenshot%20(1308).png)


# **FUTURE WORK:**
 1. The same process should be repeated for regression classifier in another.

 2. Some features can be added to the predictive factors such as item description and gender which will combine Natural Language Processing(NLP) to our image analysis.

 3. A nice deployment of predictive model as a web service needs to be done to complete our final touches on this project.
