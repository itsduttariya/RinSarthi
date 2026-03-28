import pandas as pd

# Load dataset
df = pd.read_csv("DATA/train.csv")

# Show first 5 rows
print(df.head())

# Show shape
print("Shape:", df.shape)

# Show column names
print("Columns:", df.columns)

df.columns = df.columns.str.strip()



# Info about dataset
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())



# Fill missing values

# Categorical columns → mode
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Numerical columns → mean
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].mean())


# Missing Values
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())




from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Encode all categorical columns
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print(df.head())


# Split Dataset
from sklearn.model_selection import train_test_split

# Features (everything except Loan_Status)
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)

# Target (what we want to predict)
y = df["Loan_Status"]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check shapes
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)



# Training both (Logistic Regression n Random Forest) Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize models
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()

# Train models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# Accuracy
lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("Logistic Regression Accuracy:", lr_acc)
print("Random Forest Accuracy:", rf_acc)



# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix

# Logistic Regression Evaluation
print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))


# Random Forest Evaluation
print("\nRandom Forest Report:")
print(classification_report(y_test, rf_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))




# Choosing Logistic Regression (BEST - performed better than random forest)
import pickle

# Save the best model (Logistic Regression)
pickle.dump(lr, open("MODEL/model.pkl", "wb"))
