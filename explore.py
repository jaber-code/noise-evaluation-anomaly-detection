import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def explore_dataset(dataset_name: str):
    # 1. Load the data
    print(f"--- Loading {dataset_name} ---")
    data = np.load(f"data/{dataset_name}.npz", allow_pickle=True)
    X, y = data['X'], data['y']
    
    # 2. View as a Tabular DataFrame
    # We create generic column names (Feature_0, Feature_1, etc.)
    feature_columns = [f"Feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_columns)
    df['Label'] = y # Add the label column at the end (0 = Normal, 1 = Anomaly)
    
    print("\nDataset Shape:", df.shape)
    print(f"Normal Samples: {sum(y == 0)}")
    print(f"Anomaly Samples: {sum(y == 1)}")
    
    print("\nFirst 5 Rows of the Dataset:")
    print(df.head())
    
    # 3. Visualizing the Data (PCA)
    # Since Cardio has 21 features, we can't plot it in 21D space.
    # PCA squishes those 21 features down to 2 coordinates (X and Y) so we can plot it.
    print("\nGenerating 2D Scatter Plot...")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    
    # Plot Normal data (Blue, label 0)
    plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], 
                c='blue', alpha=0.5, label='Normal (0)', s=20)
    
    # Plot Anomalous data (Red, label 1)
    plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], 
                c='red', marker='x', label='Anomaly (1)', s=50)
    
    plt.title(f"2D PCA Projection of {dataset_name}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the image so you can open it
    image_name = f"{dataset_name}_plot.png"
    plt.savefig(image_name)
    print(f"Plot saved as {image_name}! Open it to see the anomalies.")

if __name__ == "__main__":
    explore_dataset("6_cardio")