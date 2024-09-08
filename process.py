import numpy as np

from utils import z_sorce_normalize_arrays
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset


def load_file(path='', symbol='MCI'):
    if symbol == 'MCI':
        adj = np.load(path + '/NC_MCI/' + 'NC_MCI_np_dti_as_struct.npy')
        features = np.load(path + '/NC_MCI/' + 'NC_MCI_np_fmri_as_feature.npy')
        labels = np.load(path + '/NC_MCI/' + 'NC_MCI_np_corresponding_graph_tags.npy')
    elif symbol == 'EMCI':
        adj = np.load(path + '/NC_EMCI/' + 'NC_EMCI_np_dti_as_struct.npy')
        features = np.load(path + '/NC_EMCI/' + 'NC_EMCI_np_fmri_as_feature.npy')
        labels = np.load(path + '/NC_EMCI/' + 'NC_EMCI_np_corresponding_graph_tags.npy')
    elif symbol == 'LMCI':
        adj = np.load(path + '/NC_LMCI/' + 'NC_LMCI_np_dti_as_struct.npy')
        features = np.load(path + '/NC_LMCI/' + 'NC_LMCI_np_fmri_as_feature.npy')
        labels = np.load(path + '/NC_LMCI/' + 'NC_LMCI_np_corresponding_graph_tags.npy')
    elif symbol == 'AD':
        adj = np.load(path + '/NC_AD/' + 'NC_AD_np_dti_as_struct.npy')
        features = np.load(path + '/NC_AD/' + 'NC_AD_np_fmri_as_feature.npy')
        labels = np.load(path + '/NC_AD/' + 'NC_AD_np_corresponding_graph_tags.npy')
    elif symbol == 'ELMCI':
        adj = np.load(path + '/EMCI_LMCI/' + 'EMCI_LMCI_np_dti_as_struct.npy')
        features = np.load(path + '/EMCI_LMCI/' + 'EMCI_LMCI_np_fmri_as_feature.npy')
        labels = np.load(path + '/EMCI_LMCI/' + 'EMCI_LMCI_np_corresponding_graph_tags.npy')
    else:
        adj = np.load(path + '/NC_MCI_EMCI_LMCI/' + 'NC_MCI_EMCI_LMCI_np_dti_as_struct.npy')
        features = np.load(path + '/NC_MCI_EMCI_LMCI/' + 'NC_MCI_EMCI_LMCI_np_fmri_as_feature.npy')
        labels = np.load(path + '/NC_MCI_EMCI_LMCI/' + 'NC_MCI_EMCI_LMCI_np_corresponding_graph_tags.npy')

    return adj, features, labels


def load_dataset(dataset: str, balence_weights: bool = False, normalized=None, symbol='MCI'):
    adj, features, labels = load_file(dataset, symbol)
    # z-score
    features = z_sorce_normalize_arrays(features)
    features = np.transpose(features, (0, 2, 1))
    #features = PCA(features)
    if is_has_nan(labels) or is_has_nan(features) or is_has_nan(adj):
        raise ValueError("Encountered NaN value form labels or features or adj.")

    # 转化为对称矩阵
    adj = to_sys_matrix(adj)
    if balence_weights:
        adj = balance_weights_to_ones(adj)
    if normalized == 'sys':
        adj = sys_normalized_adjacency(adj)
    elif normalized == 'rw':
        adj = rw_normalized_adjacency(adj)

    return adj.astype(np.float32), features.astype(np.float32), labels.astype(np.float32)


def cross_validation(label, k_fold):
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=1)
    train_index_list, test_index_list = [], []
    for train_index, test_index in skf.split(np.zeros(len(label)), label):
        train_index_list.append(train_index)
        test_index_list.append(test_index)

    return train_index_list, test_index_list


def is_has_nan(matrix: np.ndarray) -> bool:
    # 判断矩阵中是否存在nan值
    if np.isnan(matrix).any():
        return True
    else:
        return False


def balance_weights_to_ones(matrix: np.array) -> np.ndarray:
    """
    设置所有非0元素为1，彻底转为无权图
    :param matrix:
    :return:
    """
    matrix[matrix != 0] = 1
    return matrix


def to_sys_matrix(adj: np.ndarray) -> np.ndarray:
    # 转化为对阵矩阵
    adj = adj + adj.transpose(0, 2, 1) * (adj.transpose(0, 2, 1) > adj) - adj * (adj.transpose(0, 2, 1) > adj)
    return adj


def sys_normalized_adjacency(adj: np.ndarray, is_self_edge: bool = True) -> np.ndarray:
    """
    对称归一化，D^{-1/2} Adj  D^{-1/2}
    :param adj:
    :return:
    """
    if is_self_edge:
        adj = add_self_loops(adj)
    for idx in range(adj.shape[0]):
        row_sum = adj[idx].sum(axis=1)
        D_inv = np.power(row_sum, -0.5).flatten()
        D_inv[np.isinf(D_inv)] = 0.
        D_diag = np.diag(D_inv)
        adj[idx] = D_diag @ adj[idx] @ D_diag
    return adj


def rw_normalized_adjacency(adj, is_self_edge: bool = True):
    """
    随机游走归一化邻接矩阵
    D^{-1} Adj  D^{-1}
    :param adj: 邻接矩阵
    :return: 归一化后邻接矩阵
    """
    if is_self_edge:
        adj = add_self_loops(adj)
    for idx in range(adj.shape[0]):
        row_sum = adj[idx].sum(axis=1)
        D_inv = np.power(row_sum, -1).flatten()
        D_inv[np.isinf(D_inv)] = 0.
        D_diag = np.diag(D_inv)
        adj[idx] = D_diag @ adj[idx]
    return adj


def add_self_loops(adj: np.ndarray) -> np.ndarray:
    """
    矩阵添加自环
    :param adj:
    :return:
    """
    identity_matrix = np.eye(adj.shape[1])
    return adj + identity_matrix


class CustomDataset(Dataset):
    def __init__(self, adj, features, labels):
        self.adj = adj
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.adj[index], self.features[index], self.labels[index]
