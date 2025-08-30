import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# File name
filename = "crop_prices.csv"

# ✅ Step 1: Create CSV if it doesn't exist or is empty
if not os.path.exists(filename) or os.stat(filename).st_size == 0:
    print("⚠️ CSV file not found or empty. Creating sample dataset...")
    data = {
        "Year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "Price": [1500, 1600, 1700, 1800, 1900, 2100, 2200, 2300, 2400, 2500]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
else:
    df = pd.read_csv(filename)

print("✅ Data Loaded:")
print(df.head())

# ✅ Step 2: Split into features (X) and target (y)
X = df[["Year"]]
y = df["Price"]

# ✅ Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Step 5: Predict
y_pred = model.predict(X_test)

# ✅ Step 6: Plot predictions vs actual
plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted Prices")
plt.xlabel("Year")
plt.ylabel("Price")
plt.title("Market Price Prediction")
plt.legend()
plt.show()
