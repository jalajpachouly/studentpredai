import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------------------------------
# 1. Load Data
# --------------------------------------------------------------------------
# Replace "your_data.csv" with the actual path/filename of your CSV.
df = pd.read_csv("student-dataset.csv")

# --------------------------------------------------------------------------
# 2. Basic Data Cleaning
# --------------------------------------------------------------------------
columns_to_drop = ["timestamp", "email", "student_name"]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Handle missing values
df.fillna(method="ffill", inplace=True)  # Forward fill as an example

# --------------------------------------------------------------------------
# 3. Separate Numerical and Categorical Columns
# --------------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# --------------------------------------------------------------------------
# 4. Encode Categorical Columns
# --------------------------------------------------------------------------
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# --------------------------------------------------------------------------
# 5. Scale the Data
# --------------------------------------------------------------------------
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# --------------------------------------------------------------------------
# 6. Apply PCA
# --------------------------------------------------------------------------
# Initialize PCA - adjust n_components as needed
pca = PCA(n_components=5)  # For example, retaining 5 principal components
pca_components = pca.fit_transform(df_scaled)

# Create a DataFrame for principal components
pca_df = pd.DataFrame(data=pca_components,
                      columns=[f'PC{i+1}' for i in range(pca.n_components_)])

# --------------------------------------------------------------------------
# 7. Integrate Principal Components into the Original Dataset
# --------------------------------------------------------------------------
# Option 1: Append principal components to the original dataset
updated_df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)

# Option 2: Replace the original features with principal components (if preferred)
# updated_df = pca_df.copy()

# --------------------------------------------------------------------------
# 8. Save the Updated Dataset
# --------------------------------------------------------------------------
updated_df.to_csv("updated_dataset_with_PCs.csv", index=False)
print("Updated dataset with principal components saved as 'updated_dataset_with_PCs.csv'.")

# --------------------------------------------------------------------------
# 9. (Optional) Visualize Explained Variance
# --------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# --------------------------------------------------------------------------
# 10. Practical Usage: Machine Learning Example
# --------------------------------------------------------------------------
# Assume you have a target variable for classification, e.g., 'placement_status'
# Ensure 'placement_status' exists and is encoded appropriately
if 'placement_status' in df.columns:
    # Define features and target
    X = pca_df  # Using principal components as features
    y = df['placement_status']  # Replace with your actual target column

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Initialize and train a classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # (Optional) Visualize the decision boundary using the first two PCs
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='PC1', y='PC2', hue=y_test, palette='viridis', alpha=0.7)
    plt.title('Decision Boundary using PCA Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Placement Status')
    plt.show()

else:
    print("The target variable 'placement_status' was not found in the dataset.")
    print("Please ensure you have a target variable for supervised learning tasks.")