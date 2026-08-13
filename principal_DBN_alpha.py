import numpy as np
from principal_RBM_alpha import RBM


class DBN:
    def __init__(self, network_size: list) -> None:
        self.layers = self.init_DBN(network_size)

    def init_DBN(self, network_size: list) -> list:
        """init_DBN initializes the layers of the DBN based on the specified network size."""
        dbn = []
        for i in range(len(network_size) - 1):
            p = network_size[i]
            q = network_size[i+1]
            dbn.append(RBM(p, q))
        return dbn


    def train_DBN(self, data: np.ndarray, epochs: int, lr: float, batch_size: int, verbose: bool = False) -> 'DBN':
        """Trains the DBN layer by layer using the provided training data.
        data: training data (n_samples, p)
        epochs: number of training epochs
        lr: learning rate
        batch_size: size of each mini-batch
        verbose: whether to print training progress
        """
        current_input = np.array(data, dtype=np.float32)
        for i, rbm in enumerate(self.layers):
            assert isinstance(rbm, RBM), f"Layer {i} is not an instance of RBM"

            if verbose:
                print(f"\n--- Pre-training DBN Layer {i+1}/{len(self.layers)} ---")
            
            rbm.train_RBM(current_input, epochs, lr, batch_size, verbose)
            current_input = rbm.entree_sortie_RBM(current_input)
        return self


    def generer_image_DBN(self, nb_images: int, iter_gibbs: int, img_h: int = 20, img_w: int = 16, binarize: bool = True, display: bool = True) -> np.ndarray:
        """Generates new images by performing Gibbs sampling on the top layer and then propagating down through the layers.
        nb_images: number of images to generate
        iter_gibbs: number of Gibbs sampling iterations to perform on the top layer
        img_h: height of each generated image
        img_w: width of each generated image
        binarize: whether to binarize the generated images
        display: whether to display the generated images
        """
        top_rbm = self.layers[-1]
        assert isinstance(top_rbm, RBM), "Top layer is not an instance of RBM"
        p_top = np.size(top_rbm.a, 0)
        generated_images = []

        v = (np.random.rand(nb_images, p_top) < np.random.rand(p_top)).astype(np.float32)
            
        for _ in range(iter_gibbs):
            ph_v = top_rbm.entree_sortie_RBM(v)
            h = (np.random.rand(nb_images, top_rbm.q) < ph_v).astype(np.float32) # Sample hidden layer
            
            pv_h = top_rbm.sortie_entree_RBM(h)
            v = (np.random.rand(nb_images, p_top) < pv_h).astype(np.float32) # Sample visible layer

        for j in range(len(self.layers) - 2, -1, -1): # Propagate down through the layers
            rbm_j = self.layers[j]
            assert isinstance(rbm_j, RBM), f"Layer {j} is not an instance of RBM"
            pv_h = rbm_j.sortie_entree_RBM(v)

            if binarize:
                v = (np.random.rand(nb_images, rbm_j.p) < pv_h).astype(np.float32) # Sample visible layer for next RBM
            else:
                v = pv_h  # Use probabilities directly without sampling for smoother images
        
        generated_images = v
        
        if display:
            top_rbm.display_image_RBM(generated_images, nb_images, img_h, img_w)
        
        return generated_images


if __name__ == "__main__":
    from principal_RBM_alpha import RBM
    
    # Load data
    FILE_PATH = "Data/binaryalphadigs.mat"
    CHARS = [10, 11, 12]
    data = RBM.lire_alpha_digit(FILE_PATH, CHARS)
    
    # Hyperparameters
    NETWORK_SIZE = [320, 400, 400]
    EPOCHS = 2000
    LR = 0.01
    BATCH_SIZE = 10
    
    # Train DBN
    dbn = DBN(NETWORK_SIZE)
    dbn.train_DBN(data, EPOCHS, LR, BATCH_SIZE)
    dbn.generer_image_DBN(iter_gibbs=1000, nb_images=10, display=True)