import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm


class RBM:
    def __init__(self, p, q):
        """
        p: number of visible units
        q: number of hidden units
        """
        self.p = p
        self.q = q
        params = self.init_RBM(p, q)
        self.W = params['W']
        self.a = params['a']
        self.b = params['b']

    @staticmethod
    def lire_alpha_digit(file_path: str, chars: list) -> np.ndarray:
        """Reads the specified characters from the .mat file and returns them as a numpy array.
        file_path: path to the .mat file
        chars: list of characters to read (e.g., [10, 11, 12] for 'A', 'B', 'C' or ['A', 'B', 'C'])
        """
        mat_data = scipy.io.loadmat(file_path)
        data_raw = mat_data['dat']
        
        indices = []
        for c in chars:
            if isinstance(c, int):
                indices.append(c)
            else:
                c = str(c).upper()
                if c.isdigit():
                    indices.append(ord(c) - ord('0'))  # Convert '0'-'9' to 0-9
                elif c.isalpha():
                    indices.append(ord(c) - ord('A') + 10)  # Convert 'A'-'Z' to 10-35
                else:
                    raise ValueError(f"Invalid character: {c}. Must be a digit or uppercase letter.")
        
        selected_data = []
        for idx in indices:
            char_samples = data_raw[idx]
            for sample in char_samples:
                selected_data.append(sample.flatten()) # Flatten each 20x16 image to a 320-dimensional vector

        return np.array(selected_data)


    def init_RBM(self, p: int, q: int) -> dict:
        """
        Constructs and initializes the weights and biases of an RBM.
        p: number of visible units (input dimensions)
        q: number of hidden units
        """
        rbm = {
            'W': np.random.normal(0, 0.1, (p, q)).astype(np.float32),
            'a': np.zeros(p).astype(np.float32),
            'b': np.zeros(q).astype(np.float32)
        }
        return rbm


    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        res = 1 / (1 + np.exp(-x))
        return res.astype(np.float32)


    def entree_sortie_RBM(self, v: np.ndarray) -> np.ndarray:
        """
        v: visible layer activations
        Computes P(h=1|v). Positive phase / Data representation.
        """
        activation = v @ self.W + self.b
        return self.sigmoid(activation)


    def sortie_entree_RBM(self, h: np.ndarray) -> np.ndarray:
        """
        h: hidden layer activations
        Computes P(v=1|h). Negative phase / Reconstruction.
        """
        activation = h @ self.W.T + self.a
        return self.sigmoid(activation)


    def train_RBM(self, data: np.ndarray, epochs: int, lr: float, batch_size: int, verbose: bool = False) -> 'RBM':
        """Trains the RBM using Contrastive Divergence.
        data: training data (n_samples, p)
        epochs: number of training epochs
        lr: learning rate
        batch_size: size of each mini-batch
        verbose: whether to print training progress
        """
        X = data.astype(np.float32)  # Ensure data is float32 and avoid modifying original data
        n_samples = np.size(X, 0)

        epoch_iterator = tqdm(range(epochs), desc="Training RBM", unit="epoch") if verbose else range(epochs)

        for epoch in epoch_iterator:
            # ensure batchs are not the same at each epoch
            permutation = np.random.permutation(n_samples)
            X = X[permutation]
        
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:min(i+batch_size, n_samples)]
                t_batch = np.size(X_batch, 0)
                v0 = X_batch # (t_batch, p)
            
                ph_v0 = self.entree_sortie_RBM(v0) # (t_batch, q)
                h0 = (np.random.rand(*ph_v0.shape) < ph_v0).astype(np.float32) # (t_batch, q)
            
                pv_h0 = self.sortie_entree_RBM(h0) # (t_batch, p)
                v1 = (np.random.rand(*pv_h0.shape) < pv_h0).astype(np.float32) # (t_batch, p)
                ph_v1 = self.entree_sortie_RBM(v1)
            
                da = np.sum(v0 - v1, axis=0)
                db = np.sum(ph_v0 - ph_v1, axis=0)
                dW = (v0.T @ ph_v0 - v1.T @ ph_v1)

                lr_batch = np.float32(lr / t_batch)
                self.W += lr_batch * dW
                self.a += lr_batch * da
                self.b += lr_batch * db
            
            if ((epoch + 1) % 10 == 0 or epoch == 0) and verbose:
                H = self.entree_sortie_RBM(X)
                X_rec = self.sortie_entree_RBM(H)
                error_rec = np.mean((X - X_rec) ** 2)
                epoch_iterator.set_postfix({"Reconstruction Error (Updated only every 10 epochs)": f"{error_rec:.4f}"})

        return self
    

    def display_image_RBM(self, images: list, nb_images: int, img_h: int = 20, img_w: int = 16) -> None:
        """Displays a grid of generated images from the RBM."""
        cols = min(min(5, nb_images), len(images))
        rows = int(np.ceil(nb_images / cols))

        plt.figure(figsize=(cols * 1.0, rows * 1.2)) # 1 inch per column, 1.2 inch per row

        for i in range(nb_images):
            image = np.array(images[i]).reshape(img_h, img_w)
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            
        plt.suptitle("Generated Images", fontsize=10)
        plt.tight_layout()
        plt.show()


    def generer_image_RBM(self, nb_images: int, iter_gibbs: int, display: bool = True) -> np.ndarray:
        """Generates new images using Gibbs sampling from the trained RBM.
        iter_gibbs: number of Gibbs sampling iterations
        nb_images: number of images to generate
        display: whether to display the generated images
        """
        p = np.size(self.a, 0)
        q = np.size(self.b, 0)

        v = (np.random.rand(nb_images, p) < np.random.rand(p)).astype(np.float32)  # Random binary vector of size p
            
        for _ in range(iter_gibbs):
            h = (np.random.rand(nb_images, q) < self.entree_sortie_RBM(v)).astype(np.float32) # (nb_images, q)
            v = (np.random.rand(nb_images, p) < self.sortie_entree_RBM(h)).astype(np.float32) # (nb_images, p)
        
        generated_images = v
        
        if display:
            self.display_image_RBM(generated_images, nb_images)
        
        return generated_images


if __name__ == "__main__":
    # Test parameters
    FILE_PATH = "data/binaryalphadigs.mat"
    CHARS = [10, 11, 12] # A, B, C
    P = 320
    Q = 100
    EPOCHS = 100
    LR = 0.1
    BATCH_SIZE = 10
    
    # Load data
    data = RBM.lire_alpha_digit(FILE_PATH, CHARS)
    
    # Initialize and Train
    rbm = RBM(P, Q)
    rbm.train_RBM(data, EPOCHS, LR, BATCH_SIZE)
    
    # Generate images
    rbm.generer_image_RBM(nb_images=10, iter_gibbs=1000)