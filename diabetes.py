#%%
import numpy as np 
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt

# %%
data = datasets.load_diabetes()
X = data.get('data')
y = data.get('target')
print(X.shape)
print(y.shape)

#%%
# Normalizing acording to the Dataset description
# It is already normalized, doing it again for demonstration
mean = np.mean(X, axis=0, keepdims=True)
std = np.std(X, axis=0, keepdims=True)

X = X - mean
X = X / std
sumsquare = np.sqrt(np.sum(np.power(X, 2), axis=0, keepdims=True))
X = X / sumsquare


# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %%

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

pred_train = linear_regression.predict(X_train)
pred_test = linear_regression.predict(X_test)

train_MAE = mean_absolute_error(y_train, pred_train)
test_MAE = mean_absolute_error(y_test, pred_test)

train_MSE = mean_squared_error(y_train, pred_train)
test_MSE = mean_squared_error(y_test, pred_test)

train_RMSE = root_mean_squared_error(y_train, pred_train)
test_RMSE = root_mean_squared_error(y_test, pred_test)

answer = f"""
Linear Regression
train:
    MAE: {train_MAE}
    MSE: {train_MSE}
    RMSE: {train_RMSE}
test:
    MAE: {test_MAE}
    MSE: {test_MSE}
    RMSE: {test_RMSE}
"""
print(answer)

# %%

ridge_regression = RidgeCV()
ridge_regression.fit(X_train, y_train)

pred_train = ridge_regression.predict(X_train)
pred_test = ridge_regression.predict(X_test)

train_MAE = mean_absolute_error(y_train, pred_train)
test_MAE = mean_absolute_error(y_test, pred_test)

train_MSE = mean_squared_error(y_train, pred_train)
test_MSE = mean_squared_error(y_test, pred_test)

train_RMSE = root_mean_squared_error(y_train, pred_train)
test_RMSE = root_mean_squared_error(y_test, pred_test)

answer = f"""
Ridge Regression
train:
    MAE: {train_MAE}
    MSE: {train_MSE}
    RMSE: {train_RMSE}
test:
    MAE: {test_MAE}
    MSE: {test_MSE}
    RMSE: {test_RMSE}
"""
print(answer)
# %%

lasso_regression = LassoCV()
lasso_regression.fit(X_train, y_train)

pred_train = lasso_regression.predict(X_train)
pred_test = lasso_regression.predict(X_test)

train_MAE = mean_absolute_error(y_train, pred_train)
test_MAE = mean_absolute_error(y_test, pred_test)

train_MSE = mean_squared_error(y_train, pred_train)
test_MSE = mean_squared_error(y_test, pred_test)

train_RMSE = root_mean_squared_error(y_train, pred_train)
test_RMSE = root_mean_squared_error(y_test, pred_test)

answer = f"""
Lasso Regression
train:
    MAE: {train_MAE}
    MSE: {train_MSE}
    RMSE: {train_RMSE}
test:
    MAE: {test_MAE}
    MSE: {test_MSE}
    RMSE: {test_RMSE}
"""
print(answer)

# %%
