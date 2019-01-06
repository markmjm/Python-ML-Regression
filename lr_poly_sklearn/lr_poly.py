import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


#
#############################################################################
# Load data
boston = datasets.load_boston()

print(boston.data.shape, boston.target.shape)
print(boston.feature_names)
data = pd.DataFrame(boston.data,columns=boston.feature_names)
data = pd.concat([data,pd.Series(boston.target,name='MEDV')],axis=1)
print(data.head())
print(boston.DESCR)
#
#“LSTAT” vs % of the population in the neighborhood which comes under lower economic status.
X = data[['LSTAT']]
y = data['MEDV']
plt.scatter(X, y)
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()
#
#Train test split
x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(X,y,test_size=0.10,
                                                                          random_state=42,
                                                                          shuffle=True)

# Polynomial Regression-nth order
plt.scatter(x_test_set, y_test_set, s=10, alpha=0.7, c='r')
plt.scatter(x_training_set, y_training_set, s=10, alpha=0.1, c='b' )
plt.show()
#
#visualize predictions / R2 as we inclrease the poly order
for degree in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_training_set,y_training_set)
    y_plot = model.predict(x_test_set)
    plt.scatter(x_test_set, y_plot, label="degree %d" % degree
             +'; $R^2$: %.2f' % model.score(x_test_set, y_test_set),  alpha=0.7, c='b')
    plt.scatter(X, y,  alpha=0.1, c='g', label='training data')
    plt.legend(loc='upper right')
    plt.xlabel("Test LSTAT Data")
    plt.ylabel("Predicted Price")
    plt.title("Variance Explained with Varying Polynomial")
    plt.show()
