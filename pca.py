import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------
# 1. Load Data
#
# Replace "your_data.csv" with the actual path/filename of your CSV.
# Make sure it contains all 1098 records (the snippet you showed is only a sample).
# --------------------------------------------------------------------------
df = pd.read_csv("student-dataset.csv")

# --------------------------------------------------------------------------
# 2. Basic Data Cleaning
#
#   - Drop columns that are unlikely to aid meaningful PCA, such as:
#     "timestamp", "email", "student_name" (unique identifiers).
#   - Confirm which columns you want to keep or remove based on your needs.
#   - Handle missing values if any (e.g., fill them, drop them, etc.).
# --------------------------------------------------------------------------
columns_to_drop = ["timestamp", "email", "student_name"]
for col in columns_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# (Optional) Fill or drop missing values (customize this to your dataset demands)
df.fillna(method="ffill", inplace=True)  # Forward fill just as an example

# --------------------------------------------------------------------------
# 3. Separate Numerical and Categorical Columns
#
#   - PCA works on numerical data. We need to encode categorical features.
#   - Identify which columns are numeric vs. categorical.
# --------------------------------------------------------------------------
numeric_cols = []
categorical_cols = []

for col in df.columns:
    # If it's numeric, we add it to numeric_cols; otherwise, to categorical_cols
    if pd.api.types.is_numeric_dtype(df[col]):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

# --------------------------------------------------------------------------
# 4. Encode Categorical Columns
#
#   - Convert string-based categories into integer-based categories using
#     LabelEncoder.
# --------------------------------------------------------------------------
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Now df should be entirely numeric.
# --------------------------------------------------------------------------
# 5. Scale the Data
#
#   - Standardizing features before PCA helps ensure features with large ranges
#     do not dominate those with smaller ranges.
# --------------------------------------------------------------------------
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# --------------------------------------------------------------------------
# 6. Apply PCA
#
#   - We'll choose 2 components for a simple scatterplot visualization.
#   - You can pick more components if you want to analyze the explained variance.
# --------------------------------------------------------------------------
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_scaled)

# Extract the two principal components for plotting
pc1 = pca_components[:, 0]
pc2 = pca_components[:, 1]

# Print the explained variance ratio of the two components
print("Explained variance ratio:", pca.explained_variance_ratio_)

# --------------------------------------------------------------------------
# 7. Visualize the First Two Principal Components
#
#   - We'll make a scatter plot. Optionally, pick a categorical column (e.g., 'gender')
#     to color the points.
#   - If you’d like to color by gender, ensure that 'gender' is label-encoded first.
# --------------------------------------------------------------------------

# Example: If "gender" was originally in your dataset and you want to color by it,
# ensure that "gender" is in df columns and was label-encoded. Then:
# c=df["gender"] (assuming it was label-encoded).
# Here we'll just do a simple scatter without color-coding. Feel free to adapt.
plt.figure(figsize=(8, 6))
plt.scatter(pc1, pc2, alpha=0.7, edgecolors='k')
plt.title("PCA of Student Dataset (First 2 Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

# --------------------------------------------------------------------------
# 8. (Optional) Further Analysis
#
#   - Inspect loadings (pca.components_) to see which original features
#     contribute most to each principal component.
#   - Create additional plots or a biplot for deeper analysis.
# --------------------------------------------------------------------------