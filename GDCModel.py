import numpy as np
import yaml
from scipy.linalg import expm


class PPRDataset:
    """
    Dataset preprocessed with GDC using PPR diffusion.
    Note that this implementations is not scalable
    since we directly invert the adjacency matrix.
    """

    def __init__(self,
                 alpha: float = 0.1,
                 k: int = 16,
                 eps: float = None):
        self.alpha = alpha
        self.k = k
        self.eps = eps

        super(PPRDataset, self).__init__()

    def process(self, adj):
        # obtain exact PPR matrix
        ppr_matrix = self.get_ppr_matrix(adj,
                                    alpha=self.alpha)
        if self.k:
            ppr_matrix = get_top_k_matrix(ppr_matrix, k=self.k)
        elif self.eps:
            ppr_matrix = get_clipped_matrix(ppr_matrix, eps=self.eps)
        else:
            raise ValueError
        return ppr_matrix

    def get_ppr_matrix(self,
            adj_matrix: np.ndarray,
            alpha: float = 0.1) -> np.ndarray:
        num_nodes = adj_matrix.shape[0]
        A_tilde = adj_matrix + np.eye(num_nodes)
        D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
        H = D_tilde @ A_tilde @ D_tilde
        return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)

class HeatDataset():
    """
    Dataset preprocessed with GDC using heat kernel diffusion.
    Note that this implementations is not scalable
    since we directly calculate the matrix exponential
    of the adjacency matrix.
    """
    def __init__(self,
                 t: float = 5.0,
                 k: int = 16,
                 eps: float = None):
        self.t = t
        self.k = k
        self.eps = eps

        super(HeatDataset, self).__init__()


    def process(self,adj):
        # get heat matrix as described in Berberidis et al., 2019
        heat_matrix = self.get_heat_matrix(adj, self.t)
        if self.k:
            heat_matrix = get_top_k_matrix(heat_matrix, k=self.k)
        elif self.eps:
            heat_matrix = get_clipped_matrix(heat_matrix, eps=self.eps)
        else:
            raise ValueError
        return heat_matrix

    def get_heat_matrix(self,
            adj_matrix: np.ndarray,
            t: float = 5.0) -> np.ndarray:
        num_nodes = adj_matrix.shape[0]
        A_tilde = adj_matrix + np.eye(num_nodes)
        D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
        H = D_tilde @ A_tilde @ D_tilde

        return expm(-t * (np.eye(num_nodes) - H))

def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1  # avoid dividing by zero
    return A / norm

def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1  # avoid dividing by zero
    return A / norm



if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt

    # Sample adjacency matrix
    adj_matrix = np.array([[0, 1, 0, 1, 0],
                           [1, 0, 1, 0, 1],
                           [0, 1, 0, 1, 0],
                           [1, 0, 1, 0, 1],
                           [0, 1, 0, 1, 0]])

    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)

    # Draw the graph with edge connections
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=800, font_size=10, width=2, edge_color='gray')
    plt.show()

    with open('config.yaml', 'r') as c:
        config = yaml.safe_load(c)

    dataset = HeatDataset(
        t=config['heat']['t'],
        k=config['heat']['k'],
        eps=config['heat']['eps']
    )
    heat_matrix = dataset.process(adj_matrix)
    G = nx.from_numpy_array(adj_matrix)

    # Draw the graph with edge connections
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=800, font_size=10, width=2, edge_color='gray')
    plt.show()

    dataset = HeatDataset(
        t=config['ppr']['alpha'],
        k=config['ppr']['k'],
        eps=config['ppr']['eps']
    )
    ppr_matrix = dataset.process(adj_matrix)
    G = nx.from_numpy_array(adj_matrix)

    # Draw the graph with edge connections
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=800, font_size=10, width=2, edge_color='gray')
    plt.show()



