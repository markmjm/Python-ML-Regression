import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd

plt.close("all")

# y=a0+a1x1+a2x2+⋯
rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2., 1.])

model = LinearRegression(fit_intercept=True)
model.fit(X, y)
print(model.intercept_)
print(model.coef_)

sns.set(style="ticks", color_codes=True)

df = pd.DataFrame(X, columns = ['x1','x2', 'x3'])
df['y'] = y

sns.pairplot(df)
plt.show()

rng2 = np.random.RandomState(2)
y_predict = model.predict(X)
plt.scatter(X[:, 0], y_predict, c='r')
plt.scatter(X[:, 1], y_predict, c='g')
plt.scatter(X[:, 2], y_predict, c='b')
plt.show()
df2 =  pd.DataFrame(X, columns = ['x1','x2', 'x3'])
df2['y'] = y_predict
sns.pairplot(df2)
plt.show()

plt.scatter(y, y_predict)
plt.xlabel('y_pred')
plt.ylabel('y')
plt.show()

