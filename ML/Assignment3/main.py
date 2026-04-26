from utils import get_data, plot_metrics, normalize
from model import MultiClassSVM, PCA
from typing import Tuple, List

def get_hyperparameters() -> Tuple[float, int, List[float]]:
    # get the hyperparameters
    learning_rate = 0.01
    num_iters = 1000
    
    # C should be a list with median as 1 and 5 different values.
    C = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    return learning_rate, num_iters, C


def save_metrics_to_csv(metrics: list, filename: str = 'svm_metrics_results.csv') -> None:
    # Helper function to dump the list of tuples to a CSV file
    print(f"\nSaving metrics to {filename}...")
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write the header row first
        writer.writerow(['C_value', 'k_components', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])
        # Write all the data rows
        writer.writerows(metrics)
    print("Save complete.")


def main() -> None:
    # hyperparameters
    learning_rate, num_iters, C_list = get_hyperparameters()

    # get data
    X_train, X_test, y_train, y_test = get_data()

    # normalize the data
    X_train, X_test = normalize(X_train, X_test)

    # Define the number of principal components (k) to test
    k_values = [10, 50, 100, 200, 500]
    
    metrics = []
    
    # Outer loop: Iterate over the regularization parameter C
    for C_i in C_list:
        print(f"\n{'='*50}")
        print(f"Evaluating SVM with C = {C_i}")
        print(f"{'='*50}")
        
        # Inner loop: Iterate over the different values of k (principal components)
        for k in k_values:
            print(f"--- Running PCA and SVM for k={k} components ---")
            
            # reduce the dimensionality of the data
            pca = PCA(n_components=k)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # create a model
            svm = MultiClassSVM(num_classes=10)

            # fit the model
            svm.fit(
                X_train_pca, y_train, 
                C=C_i,
                learning_rate=learning_rate,
                num_iters=num_iters
            )

            # evaluate the model
            accuracy = svm.accuracy_score(X_test_pca, y_test)
            precision = svm.precision_score(X_test_pca, y_test)
            recall = svm.recall_score(X_test_pca, y_test)
            f1_score = svm.f1_score(X_test_pca, y_test)

            # We now store C_i in the metrics tuple as well
            metrics.append((C_i, k, accuracy, precision, recall, f1_score))

            print(f'C={C_i}, k={k:3d} | acc={accuracy:.4f}, prec={precision:.4f}, rec={recall:.4f}, f1={f1_score:.4f}')

    save_metrics_to_csv(metrics)
    # plot and save the results
    plot_metrics(metrics)

if __name__ == '__main__':
    main()