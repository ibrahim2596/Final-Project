import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

alldata = pd.read_csv('car_prediction_data.csv')
dropped_alldata =alldata.drop(columns=["Model","Car ID"])
data = dropped_alldata.rename(columns={"Year" : "Model Year","Mileage" : "Kilometers"})
data["Kilometers"]=data["Kilometers"]*1.609
if data.isna().sum().sum() > 0:
    print("Missing values detected, filling with median values.")
    data.fillna(data.median(), inplace=True)
else :
    print(data.isna().sum())
print(data.head())
Average_Price = data["Price"].median()
print(f"Average Price is : {Average_Price:.2f} EGP")

plt.figure(figsize=(10, 6))
plt.scatter(data["Kilometers"],data["Price"],color='blue', marker='o',alpha=0.5)
plt.xlabel("Kilometers (KM)")
plt.ylabel("Price (EGP)")
plt.title("Mileage Vs Price")
plt.show()

plt.bar(data["Brand"], data["Price"].median(), color='green')
plt.title("Average Car Price per Brand")
plt.xlabel("Model")
plt.ylabel("Price")
plt.show()

categorical_cols = ["Brand", "Fuel Type","Transmission","Condition","Engine Size"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(columns=["Brand"])
y = data["Brand"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.4f}")