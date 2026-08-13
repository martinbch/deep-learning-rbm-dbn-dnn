import struct
import numpy as np
from tqdm import tqdm
from principal_DBN_alpha import DBN
from principal_RBM_alpha import RBM

class DNN(DBN):
    def __init__(self, network_size: list) -> None:
        """Initializes the DNN with the specified network size."""
        self.network_size = network_size
        self.layers = self.init_DNN(network_size)


    @staticmethod
    def load_mnist(images_path: str, labels_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Loads the MNIST dataset from the specified file paths and returns the images and one-hot encoded labels.
        images_path: path to the MNIST images file
        labels_path: path to the MNIST labels file
        """
        with open(images_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
        
        images = (images > 127).astype(np.float32)

        with open(labels_path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels_raw = np.fromfile(f, dtype=np.uint8)
            
        labels_one_hot = np.zeros((num, 10))
        for i in range(num):
            labels_one_hot[i, labels_raw[i]] = 1
            
        return images, labels_one_hot


    def init_DNN(self, network_size: list) -> list:
        """Initializes the layers of the DNN based on the specified network size."""
        dbn = self.init_DBN(network_size)
        return dbn


    def pretrain_DNN(self, data: np.ndarray, epochs: int, lr: float, batch_size: int, verbose: bool = False) -> 'DNN':
        """
        Pre-trains the DNN layer by layer using unsupervised learning (RBM training).
        data: training data (n_samples, input_dim)
        epochs: number of training epochs for each RBM layer
        lr: learning rate for RBM training
        batch_size: size of each mini-batch for RBM training
        verbose: whether to print training progress for each RBM layer
        """
        classification_layer = self.layers.pop()
        self.train_DBN(data, epochs, lr, batch_size, verbose)
        self.layers.append(classification_layer)
        return self


    def calcul_softmax(self, rbm: RBM, data: np.ndarray) -> np.ndarray:
        """Compute P(h|v) for the given RBM and input data, then apply the softmax function to get class probabilities.
        rbm: the RBM representing the classification layer
        data: input data, size (n_samples, input_dim)
        output: class probabilities, size (n_samples, output_dim)
        """
        z = data @ rbm.W + rbm.b
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


    def entree_sortie_reseau(self, data: np.ndarray) -> list:
        outputs = [data]
        current_input = data
        
        for i in range(len(self.layers) - 1):
            rbm_i = self.layers[i]
            assert isinstance(rbm_i, RBM), f"Layer {i} is not an RBM"
            current_input = rbm_i.entree_sortie_RBM(current_input)
            outputs.append(current_input)
            
        final_rbm = self.layers[-1]
        probs = self.calcul_softmax(final_rbm, current_input)
        outputs.append(probs)
        
        return outputs


    def retropropagation(self, data: np.ndarray, labels: np.ndarray, epochs: int, lr: float, batch_size: int, verbose: bool = False) -> 'DNN':
        """Fine-tunes the DNN using supervised learning (backpropagation).
        data: training data (n_samples, input_dim)
        labels: one-hot encoded labels (n_samples, output_dim)
        epochs: number of training epochs
        lr: learning rate
        batch_size: size of each training batch
        verbose: whether to print training progress
        """
        X = data.astype(np.float32, copy=True)  
        Y = labels.astype(np.float32, copy=True)
        n_samples = np.size(X, 0)

        epoch_iterator = tqdm(range(epochs), desc="Backpropagation (Supervised)", unit="epoch") if verbose else range(epochs)
        
        for epoch in epoch_iterator:
            permutation = np.random.permutation(n_samples)
            data_shuffled = X[permutation]
            labels_shuffled = Y[permutation]
            
            epoch_loss = 0.0
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = data_shuffled[start_idx:end_idx]
                Y_batch = labels_shuffled[start_idx:end_idx]
                t_batch = end_idx - start_idx
                
                activations = self.entree_sortie_reseau(X_batch)  
                delta = activations[-1] - Y_batch
                lr_batch = np.float32(lr / t_batch)

                for l in range(len(self.layers) - 1, -1, -1):
                    rbm_l = self.layers[l]
                    assert isinstance(rbm_l, RBM), f"Layer {l} is not an RBM"

                    a_in = activations[l] # input to layer l, also the output of layer l-1

                    dW = (a_in.T @ delta)
                    db = np.sum(delta, axis=0)

                    if l > 0:
                        sigmoid_derivative = a_in * (1 - a_in)
                        delta = (delta @ rbm_l.W.T) * sigmoid_derivative

                    rbm_l.W -= lr_batch * dW
                    rbm_l.b -= lr_batch * db

                predictions = activations[-1]
                epsilon = 1e-15
                predictions = np.clip(predictions, epsilon, 1 - epsilon) # Avoid log(0)
                batch_loss = -np.sum(Y_batch * np.log(predictions))
                epoch_loss += batch_loss

            if verbose:
                cross_entropy = epoch_loss / n_samples
                epoch_iterator.set_postfix({"CE Loss": f"{cross_entropy:.4f}"})
            
        return self

    def test_DNN(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Evaluates the DNN on the test set and computes the error rate.
        data: test data (n_samples, input_dim)
        labels: one-hot encoded true labels (n_samples, output_dim)
        returns: error rate (float)
        """
        activations = self.entree_sortie_reseau(data)
        probabilites = activations[-1]
        
        # Convert probabilities to class predictions
        predicted_classes = np.argmax(probabilites, axis=1)
        # Convert one-hot encoded labels to class indices
        true_classes = np.argmax(labels, axis=1)
        
        error_rate = 1.0 - np.mean(predicted_classes == true_classes)
        return error_rate


if __name__ == "__main__":
    from principal_DBN_alpha import DBN
    # Assuming DNN class is defined above in this file

    # --- 1. Data Loading ---
    path_train_img = "Data/train-images-idx3-ubyte"
    path_train_lab = "Data/train-labels-idx1-ubyte"
    path_test_img  = "Data/t10k-images-idx3-ubyte"
    path_test_lab  = "Data/t10k-labels-idx1-ubyte"

    X_train, y_train = DNN.load_mnist(path_train_img, path_train_lab)
    X_test, y_test   = DNN.load_mnist(path_test_img, path_test_lab)

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # --- 2. Hyperparameters ---
    NETWORK_SIZE = [784, 200, 100, 10]
    PRETRAIN_EPOCHS = 10
    FINETUNE_EPOCHS = 50
    LR = 0.1
    BATCH_SIZE = 64

    # --- 3. Execution Pipeline ---
    print("\n--- Initializing DNN ---")
    dnn = DNN(NETWORK_SIZE)

    print("\n--- Unsupervised Pre-training (DBN Phase) ---")
    dnn.pretrain_DNN(PRETRAIN_EPOCHS, LR, BATCH_SIZE, X_train)

    print("\n--- Supervised Fine-tuning (Backpropagation) ---")
    dnn.retropropagation(FINETUNE_EPOCHS, LR, BATCH_SIZE, X_train, y_train)

    print("\n--- Model Evaluation ---")
    dnn.test_DNN(X_test, y_test)