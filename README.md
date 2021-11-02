# linear-regression-nasscom
In [ ]:
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
Data is loaded
In [ ]:
#The original data can never be added or deleted columns
original_data = pd.read_csv("fish.csv")
#The data variable is used to make modifications on it
data = copy.deepcopy(original_data)
data.head()
linkcode
Check null data items¶
In [ ]:
np.sum(data.isnull())
There aren't null items
Data is treated (Strings converted to numerical data)
The only non numerical column is the 'Species', so this one is encoded to an integer
In [ ]:
original_data["Species"] = pd.DataFrame(original_data["Species"]).apply(LabelEncoder().fit_transform)
Let's see the linear correlation of the different features
In [ ]:
sns.heatmap(data.corr(), annot=True)
In [ ]:
corr = data.corr()["Weight"].drop("Weight")
print(corr)
Error is studied according to the number of degree of the regression
The aim of this cell is to choose the best degree for the regression. So as to achive this goal:
•	It iterates over the diferent degrees.
•	For each one, the model is trained several times (50 for example). For each training the training and test error is recorded. For each training, is randomly shuffled between training and test data. The purpose of this strategy is to getting a non random error, by averaging the errors. A conclusion can be drawn from the resulting plots:
•	2 may be the most suitable degree for the regression because:
	It gets the lowest test error.
	It gets the closest test error to the training one.
	That's why a balance between bias and variance is found
In [ ]:
#Variables for keeping track of errors are initialized
e_train = []
e_test = []
e_train_hist = []
e_test_hist = []
alpha_hist = []
alpha = []

#Max degree of the regression
max_degree = 5

#No. of training times
training_times = 50

#Iterate over the different degrees
for degree in range(1,max_degree):
    poly = PolynomialFeatures(degree)
    data = copy.deepcopy(original_data)
    y = pd.DataFrame(data["Weight"])
    data = data.drop("Weight", axis = 1)
    x = poly.fit_transform(data)
    for i in range(training_times):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=np.random.randint(100))
        model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 2, 4, 6, 8, 16, 32, 40, 50, 80, 100, 150, 200, 250, 300, 350, 400])
        model.fit(x_train, y_train)
        #Training error is recorded
        e = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))
        e_train.append(e)
        #Test error is recorded
        e = np.sqrt(mean_squared_error(y_test, model.predict(x_test)))
        e_test.append(e)
        #The alpha hyperparameter is recorded
        alpha.append(model.alpha_)
    #The records of the current degree are saved
    e_train_hist.append(e_train)
    e_train = []
    e_test_hist.append(e_test)
    e_test = []
    alpha_hist.append(alpha)
    alpha = []

#The mean for each degree is calculated
e_train = np.mean(np.array(e_train_hist),axis=1)
e_test = np.mean(np.array(e_test_hist),axis=1)
alpha = np.mean(np.array(alpha_hist),axis=1)

#The errors and alpha record is plotted
plt.plot(range(1,max_degree), e_train, 'o-', label = "train")
plt.plot(range(1,max_degree), e_test, 'o-',label = "test")
plt.legend()
plt.figure()
plt.plot(range(1,max_degree), alpha, 'o-',label = "alpha")
plt.legend()
Error is studied according to amount of data
The aim of this cell is to plot the learning curve of the model.
As a result, it can be easily spotted that training a test error end up close one to each other.
In addition, the hyperparameter alpha gets bigger and bigger because overfitting is decreasing for every dataset size iteration.
In [ ]:
#Variables for keeping track of errors are initialized
e_train = []
e_test = []
e_train_hist = []
e_test_hist = []
alpha_hist = []
alpha = []

#Max degree of the regression
max_degree = 5

#No. of training times
training_times = 50

#No. of training examples
m = original_data.shape[0]

step = 1

degree = 2

#For every iteration diferent amounts of data are selected
for n_data in range(20, m, step):
    poly = PolynomialFeatures(degree)
    # The model is trained several times with diferent data so as to get a non-random and more precise error. 
    for i in range(training_times):
        data = copy.deepcopy(original_data)
        data = data.iloc[np.random.permutation(np.arange(0,m)),:] #Data is shuffled
        data = data.iloc[1:n_data,:]
        y = pd.DataFrame(data["Weight"])
        data = data.drop("Weight", axis = 1)
        x = poly.fit_transform(data)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=np.random.randint(100))
        model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 2, 4, 6, 8, 16, 32, 40, 50, 80, 100, 150, 200, 250, 300, 350, 400])
        model.fit(x_train, y_train)
        #Training error is recorded
        e = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))
        e_train.append(e)
        #Test error is recorded
        e = np.sqrt(mean_squared_error(y_test, model.predict(x_test)))
        e_test.append(e)
        #The alpha hyperparameter is recorded
        alpha.append(model.alpha_)
    #The records of the current degree are saved
    e_train_hist.append(e_train)
    e_train = []
    e_test_hist.append(e_test)
    e_test = []
    alpha_hist.append(alpha)
    alpha = []

#The mean for every training examples amount is calculated
e_train = np.mean(np.array(e_train_hist),axis=1)
e_test = np.mean(np.array(e_test_hist),axis=1)
alpha = np.mean(np.array(alpha_hist),axis=1)

#The errors and alpha record are plotted
plt.plot(range(20, m, step), e_train, 'o-', label = "train")
plt.plot(range(20, m, step), e_test, 'o-',label = "test")
plt.legend()
plt.figure()
plt.plot(range(20, m, step), alpha, 'o-',label = "alpha")
plt.legend()
