import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('startup.csv', encoding='latin1')

# Clean the column names by removing leading/trailing spaces
df.columns = df.columns.str.strip()

# Define numeric columns based on your earlier message
numeric_columns = ['funding_total_usd', 'funding_rounds', 'seed', 'venture', 
                   'equity_crowdfunding', 'angel', 'product_crowdfunding', 
                   'private_equity', 'debt_financing']

# Convert 'funding_total_usd' to numeric, handling commas and dashes
df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'].str.replace(',', '').replace(' - ', pd.NA), errors='coerce')

# Impute missing values for the numeric columns
imputer = SimpleImputer(strategy='median')  # Using median to reduce the impact of outliers
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# Now extract features and target
X = df[numeric_columns]  # Features
y = df['status']  # Target variable

# One-hot encoding for the target variable
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

# Scaling the numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.decomposition import PCA

# Check the number of features in X_scaled
print(f"Number of features in X_scaled: {X_scaled.shape[1]}")  # This should show the number of features (columns)

# Apply PCA to reduce the features to top 9 components (adjust according to the number of available features)
pca = PCA(n_components=9)  # Must be <= the number of features
X_pca = pca.fit_transform(X_scaled)

# Check how much variance is explained by each component
explained_variance = pca.explained_variance_ratio_
print("Explained variance by each component:", explained_variance)
print("Total explained variance:", sum(explained_variance))


from sklearn.model_selection import train_test_split

# Split the data into training and test sets (e.g., 80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Define a simple SVM model with a few different kernels
svm = SVC()

# Define the hyperparameters to tune
param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],    # Different kernels to try
    'C': [0.1, 1, 10, 100],                # Regularization parameter
    'gamma': ['scale', 'auto'],            # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    'degree': [3, 4, 5]                    # Degree for polynomial kernel
}

# Use GridSearchCV to search for the best combination of hyperparameters
grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train.argmax(axis=1))  # We use argmax to convert one-hot labels to single-label classes

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Predict on test set
y_pred = grid_search.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)
print(f"Accuracy on the test set: {accuracy:.4f}")

# Detailed classification report
print("Classification Report:\n", classification_report(y_test.argmax(axis=1), y_pred, target_names=encoder.categories_[0]))
