import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)

model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit);
plt.show()

# y =  2 * x - (5 + rng.randn(50))
# y = m x + b
print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)
#Predicted coef is close to 2
#Predicted intercept is close to 5