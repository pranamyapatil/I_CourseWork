import numpy as np
from tqdm import tqdm


# class SupportVectorModel:
#     def __init__(self) -> None:
#         self.w = None
#         self.b = None
    
#     def _initialize(self, X) -> None:
#         # initialize the parameters
#         pass

#     def fit(
#             self, X, y, 
#             learning_rate: float,
#             num_iters: int,
#             C: float = 1.0,
#     ) -> None:
#         self._initialize(X)
        
#         # fit the SVM model using stochastic gradient descent
#         for i in tqdm(range(1, num_iters + 1)):
#             # sample a random training example
#             raise NotImplementedError
    
#     def predict(self, X) -> np.ndarray:
#         # make predictions for the given data
#         raise NotImplementedError

#     def accuracy_score(self, X, y) -> float:
#         # compute the accuracy of the model (for debugging purposes)
#         return np.mean(self.predict(X) == y)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None

    def _initialize(self, X) -> None:
        num_features = X.shape[1]
        self.w = np.zeros(num_features)
        self.b = 0.0

    def fit(self, X, y, learning_rate: float, num_iters: int, C: float = 1.0) -> None:
        self._initialize(X)
        N = X.shape[0]

        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            idx = np.random.randint(0, N)
            x_i = X[idx]
            y_i = y[idx]

            # Calculate the condition for hinge loss margin
            # y_i(w^T x_i + b) < 1
            if y_i * (np.dot(x_i, self.w) + self.b) < 1.0:
                # Subgradient with respect to w and b
                # Note: No 'N' multiplier here based on the specific PDF objective
                dw = self.w - C * y_i * x_i
                db = -C * y_i
            else:
                dw = self.w
                db = 0.0

            # Update weights
            self.w -= learning_rate * dw
            self.b -= learning_rate * db

    def predict(self, X) -> np.ndarray:
        return np.sign(np.dot(X, self.w) + self.b)

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)

class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = [SupportVectorModel() for _ in range(self.num_classes)]

    def fit(self, X, y, **kwargs) -> None:
        # 1-vs-rest training
        for c in range(self.num_classes):
            print(f"Training SVM for class {c}...")
            # Binarize labels: 1 for the current class, -1 for all others
            y_bin = np.where(y == c, 1, -1)
            self.models[c].fit(X, y_bin, **kwargs)

    def predict(self, X) -> np.ndarray:
        # Get raw score w^Tx + b for all 10 classes
        scores = np.zeros((X.shape[0], self.num_classes))
        for c in range(self.num_classes):
            scores[:, c] = np.dot(X, self.models[c].w) + self.models[c].b
        
        # Return the class with the highest score
        return np.argmax(scores, axis=1)

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)

    def precision_score(self, X, y) -> float:
        preds = self.predict(X)
        precisions = []
        for c in range(self.num_classes):
            tp = np.sum((preds == c) & (y == c))
            fp = np.sum((preds == c) & (y != c))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return np.mean(precisions) # Macro-average

    def recall_score(self, X, y) -> float:
        preds = self.predict(X)
        recalls = []
        for c in range(self.num_classes):
            tp = np.sum((preds == c) & (y == c))
            fn = np.sum((preds != c) & (y == c))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return np.mean(recalls)

    def f1_score(self, X, y) -> float:
        prec = self.precision_score(X, y)
        rec = self.recall_score(X, y)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


# class MultiClassSVM:
#     def __init__(self, num_classes: int) -> None:
#         self.num_classes = num_classes
#         self.models = []
#         for i in range(self.num_classes):
#             self.models.append(SupportVectorModel())
    
#     def fit(self, X, y, **kwargs) -> None:
#         # first preprocess the data to make it suitable for the 1-vs-rest SVM model
#         # then train the 10 SVM models using the preprocessed data for each class
#         raise NotImplementedError

#     def predict(self, X) -> np.ndarray:
#         # pass the data through all the 10 SVM models and return the class with the highest score
#         raise NotImplementedError

#     def accuracy_score(self, X, y) -> float:
#         return np.mean(self.predict(X) == y)
    
#     def precision_score(self, X, y) -> float:
#         raise NotImplementedError
    
#     def recall_score(self, X, y) -> float:
#         raise NotImplementedError
    
#     def f1_score(self, X, y) -> float:
#         raise NotImplementedError


def my_qr(A):
    """
    Perform QR decomposition using Gram-Schmidt orthogonalization.
    Decomposes matrix A into orthogonal matrix Q and upper triangular matrix R.
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
            
        R[j, j] = np.linalg.norm(v)
        # Avoid division by zero
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = v
            
    return Q, R

def my_eigh(A, max_iter=200, tol=1e-8):
    """
    Compute eigenvalues and eigenvectors for a symmetric matrix A
    using the iterative QR algorithm.
    """
    n = A.shape[0]
    X = np.copy(A)
    # Start with identity matrix to accumulate the eigenvectors
    eigenvectors = np.eye(n) 
    
    for i in range(max_iter):
        # Step 1: Decompose current matrix X into Q and R
        Q, R = my_qr(X)
        
        # Step 2: Recombine as R * Q to form the next iteration of X
        X = np.dot(R, Q) 
        
        # Step 3: Accumulate the Q matrices to form the eigenvectors
        eigenvectors = np.dot(eigenvectors, Q) 
        
        # Step 4: Check for convergence by looking at off-diagonal elements
        off_diag_sum = np.sum(np.abs(X)) - np.sum(np.abs(np.diag(X)))
        if off_diag_sum < tol:
            break
            
    # The eigenvalues are the diagonal elements of the final X matrix
    eigenvalues = np.diag(X)
    return eigenvalues, eigenvectors


class PCAScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Step 1: Mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2: Compute the covariance matrix
        # Rowvar=False ensures columns represent variables (pixels)
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        print("Computing eigenvalues/eigenvectors from scratch. This may take a minute...")
        # Step 3: Calculate eigenvalues and eigenvectors
        # eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues, eigenvectors = my_eigh(cov_matrix)
        
        # Step 4: Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # Step 5: Select the top n_components
        self.components = sorted_eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Step 6: Project the data onto the principal components
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)