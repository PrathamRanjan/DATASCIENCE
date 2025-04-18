import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load CSV data
df = pd.read_csv("mock_data.csv")

# Basic EDA
print(df.head())
print(df.describe())
print(df.isnull().sum())

# Visualize correlation
sns.heatmap(df.corr(), annot=True)
plt.show()

# Train/test split
X = df[['feature1', 'feature2']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
print("Score:", model.score(X_test, y_test))
