import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = np.array([[1], [2], [3], [4], [5]])
x2=np.array([[60],[65],[70],[80],[90]])
y = np.array([12, 18, 25, 27, 35])  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X2_scal=scaler.fit_transform(x2)

model = LinearRegression()
model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)
print("Model Parameters:")
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("\nEvaluation Metrics:")
print("\nERROR:",y-y_pred)
print("MSE:", mean_squared_error(y, y_pred))
print("MAE:", mean_absolute_error(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
print("R2 :", r2_score(y, y_pred))
print("R2 NEW:",1-(np.sum((y-y_pred)**2)/np.sum(y-(np.mean(y)**2))))

x_new=np.array([[6]])
x_new_scaled=scaler.transform(x_new)
print("\nPrediction for 6 study hours:", model.predict(x_new_scaled)[0])
