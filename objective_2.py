import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('retractions35215.csv')

# Drop irrelevant columns
data.drop(["Record ID","Institution","Title", "Author","URLS", "RetractionDOI", "OriginalPaperDOI", "Notes", "Reason","RetractionNature","Paywalled",], axis=1, inplace=True)

# Convert date columns to datetime
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], dayfirst=True, errors='coerce')
data['OriginalPaperDate'] = pd.to_datetime(data['OriginalPaperDate'], dayfirst=True, errors='coerce')

# Drop rows with missing dates
data.dropna(subset=['RetractionDate', 'OriginalPaperDate'], inplace=True)

# Calculate period of circulation
data['PeriodOfCirculation'] = (data['RetractionDate'] - data['OriginalPaperDate']).dt.days

# Drop the original datetime columns as they are no longer needed
data.drop(['RetractionDate', 'OriginalPaperDate'], axis=1, inplace=True)

# Separate categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
print(categorical_columns)
# Apply frequency encoding to categorical variables
for col in categorical_columns:
    freq = data[col].value_counts()
    data[col] = data[col].map(freq)

# Drop rows with any remaining missing values
data.dropna(inplace=True)

# Get the response variable as ndarray
y = data['PeriodOfCirculation'].values

# Drop the response variable so what remains are the explanatory variables
data.drop("PeriodOfCirculation", axis=1, inplace=True)
print(data)
# Extract the explanatory variables as ndarray
X = data.values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Standardize the explanatory variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build, apply and evaluate Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1000, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Calculate evaluation metrics for Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = math.sqrt(mean_squared_error(y_test, y_pred_rf))

# Build, apply and evaluate Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Calculate evaluation metrics for Linear Regression
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = math.sqrt(mean_squared_error(y_test, y_pred_lr))

# Build, apply and evaluate Support Vector Regression model
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)

# Calculate evaluation metrics for Support Vector Regression
mae_svr = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = math.sqrt(mean_squared_error(y_test, y_pred_svr))

print("---Random Forest Regressor---")
print(f"MAE: {mae_rf:.2f}")
print(f"MSE: {mse_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")

print("---Linear Regression---")
print(f"MAE: {mae_lr:.2f}")
print(f"MSE: {mse_lr:.2f}")
print(f"RMSE: {rmse_lr:.2f}")

print("---Support Vector Regression---")
print(f"MAE: {mae_svr:.2f}")
print(f"MSE: {mse_svr:.2f}")
print(f"RMSE: {rmse_svr:.2f}")

# Compare the predictions of all models
print("Random Forest predictions:", y_pred_rf)
print("Linear Regression predictions:", y_pred_lr)
print("Support Vector Regression predictions:", y_pred_svr)

# Plot the actual vs predicted values
plt.figure(figsize=(18, 6))

# Plot for Random Forest
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_rf, color='blue', label='Predicted')
plt.plot(y_test, y_test, color='red', label='Actual')
plt.xlabel('Actual Period of Circulation')
plt.ylabel('Predicted Period of Circulation')
plt.title('Random Forest Regressor')
plt.legend()

# Plot for Linear Regression
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_lr, color='green', label='Predicted')
plt.plot(y_test, y_test, color='red', label='Actual')
plt.xlabel('Actual Period of Circulation')
plt.ylabel('Predicted Period of Circulation')
plt.title('Linear Regression')
plt.legend()

# Plot for Support Vector Regression
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_svr, color='purple', label='Predicted')
plt.plot(y_test, y_test, color='red', label='Actual')
plt.xlabel('Actual Period of Circulation')
plt.ylabel('Predicted Period of Circulation')
plt.title('Support Vector Regression')
plt.legend()

plt.tight_layout()
plt.show()
