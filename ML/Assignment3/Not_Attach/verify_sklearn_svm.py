import numpy as np
import torchvision
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def verify_with_sklearn() -> None:
    # 1. Load Data (Using torchvision to perfectly match your utils.py)
    print("Loading FashionMNIST dataset...")
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)
    
    X_train = train_set.data.numpy().reshape(-1, 28*28).astype(np.float32)
    y_train = train_set.targets.numpy()
    
    X_test = test_set.data.numpy().reshape(-1, 28*28).astype(np.float32)
    y_test = test_set.targets.numpy()

    # 2. Normalize Data (Z-score normalization)
    print("Normalizing data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    k_values = [10, 50, 100, 200, 500]
    C_median = 1.0 
    
    print(f"\n--- Scikit-Learn Verification (Linear SVM, C={C_median}) ---")
    
    for k in k_values:
        # 3. PCA Dimensionality Reduction
        pca = PCA(n_components=k, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # 4. Train Linear SVM (1-vs-rest is the default for LinearSVC)
        # We use dual=False because for PCA, n_samples (60,000) > n_features (k)
        clf = LinearSVC(C=C_median, max_iter=2000, dual=False, random_state=42)
        clf.fit(X_train_pca, y_train)
        
        # 5. Predict and Evaluate
        y_pred = clf.predict(X_test_pca)
        
        # Calculate macro-averaged metrics (same as your custom implementation)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        print(f"k={k:3d} | acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")

if __name__ == "__main__":
    verify_with_sklearn()