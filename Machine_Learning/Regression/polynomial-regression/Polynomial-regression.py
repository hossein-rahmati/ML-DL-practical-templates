# %% Importin libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% Importing datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# %% Training the linear regression model on the whole dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# y_pred = lin_reg.predict(X_test)
# print(y_pred)

# %% Training the polynomial regression model on the whole dataset
poly_reg = PolynomialFeatures(degree=10)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# %% visualizing the linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Truth or Bluff (Linear)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# %% visualizing the polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')
plt.title("Truth or Bluff (Polynomial)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# %% visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(
    poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# %% predicting new result with linear regression
lin_reg.predict([[6.5]])

# %% predicting new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
