import os
import numpy as np
import scipy.io as sio
from scipy import sparse
import sklearn.preprocessing as skp

def load_dataset(dataset_name, data_path, sample_size=None, data_norm='standard'):
    """
    加载并预处理指定的数据集。
    """
    data_X = []
    if dataset_name == 'Scene15':
        mat = sio.loadmat(os.path.join(data_path, 'scene15.mat'))
        X = mat['X'][0]
        num_views = mat['X'].shape[1]
        data_X = [X[i].astype('float32') for i in range(num_views)]
        label_y = np.squeeze(mat['Y'])
    elif dataset_name == 'LandUse21':
        mat = sio.loadmat(os.path.join(data_path, 'LandUse_21.mat'))
        data_X.append(sparse.csr_matrix(mat['X'][0, 1]).toarray())
        data_X.append(sparse.csr_matrix(mat['X'][0, 2]).toarray())
        label_y = np.squeeze(mat['Y']).astype('int')
    elif dataset_name == 'Reuters':
        mat = sio.loadmat(os.path.join(data_path, 'Reuters_dim10.mat'))
        data_X.append(np.vstack((mat['x_train'][0], mat['x_test'][0])))
        data_X.append(np.vstack((mat['x_train'][1], mat['x_test'][1])))
        label_y = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))
    elif dataset_name == "Hdigit":
        mat = sio.loadmat(os.path.join(data_path, 'Hdigit.mat'))
        num_views = mat['X'].shape[1]
        X = mat['X'][0]
        data_X = [X[i].astype('float32') for i in range(num_views)]
        label_y = np.array(np.squeeze(mat['Y'])).astype(np.int32)
    elif dataset_name == "cub_googlenet":
        mat = sio.loadmat(os.path.join(data_path, 'cub_googlenet_doc2vec_c10.mat'))
        num_views = mat['X'][0].shape[0]
        X = mat['X'][0]
        data_X = [X[i].astype('float32') for i in range(num_views)]
        label_y = np.squeeze(mat['gt'])
    elif dataset_name == "RGB-D":
        mat = sio.loadmat(os.path.join(data_path, 'RGB-D.mat'))
        num_views = mat['X'].shape[1]
        X = mat['X'][0]
        data_X = [X[i].astype('float32') for i in range(num_views)]
        label_y = np.squeeze(mat['Y'])
    elif dataset_name == "Cora":
        mat = sio.loadmat(os.path.join(data_path, 'Cora.mat'))
        num_views = mat['X'].shape[1]
        X = mat['X'][0]
        data_X = [X[i].astype('float32') for i in range(num_views)]
        label_y = np.squeeze(mat['Y'])
    elif dataset_name == "ALOI":
        mat = sio.loadmat(os.path.join(data_path, 'ALOI-100.mat'))
        num_views = mat['X'].shape[1]
        X = mat['X'][0]
        data_X = [X[i].astype('float32') for i in range(num_views)]
        label_y = np.squeeze(mat['Y'])
    elif dataset_name == 'Caltech101':
        mat = sio.loadmat(os.path.join(data_path, '2view-caltech101-8677sample.mat'))
        num_views = mat['X'].shape[1]
        X = mat['X'][0]
        data_X = [X[i].T.astype('float32') for i in range(num_views)]
        label_y = np.squeeze(mat['gt']) - 1
    elif dataset_name == "NUSWIDE":
        mat = sio.loadmat(os.path.join(data_path, "nuswide_deep_2_view.mat"))
        data_X.append(mat["Img"])
        data_X.append(mat["Txt"])
        label_y = np.squeeze(mat["label"].T)
    elif dataset_name == "CCV":
        mat = sio.loadmat(os.path.join(data_path, 'CCV.mat'))
        num_views = mat['X'].shape[1]
        X = mat['X'][0]
        data_X = [X[i].astype('float32') for i in range(num_views)]
        label_y = np.squeeze(mat['Y'])
    elif dataset_name == "SUNRGBD":
        mat = sio.loadmat(os.path.join(data_path, 'SUNRGBD_fea.mat'))
        num_views = mat['X'].shape[1]
        X = mat['X'][0]
        data_X = [X[i].astype('float32') for i in range(num_views)]
        label_y = np.squeeze(mat['Y'])
    elif dataset_name == "cifar100":
        mat = sio.loadmat(os.path.join(data_path, 'cifar100.mat'))
        num_views = mat['X'][0].shape[0]
        X = mat['X'][0]
        data_X = [X[i].astype('float32') for i in range(num_views)]
        label_y = np.squeeze(mat['Y'])
    elif dataset_name == "YouTubeFace":
        mat = sio.loadmat(os.path.join(data_path, 'YouTubeFace.mat'))
        num_views = mat['X'].shape[0]
        X = mat['X']
        data_X = [X[i][0].astype('float32') for i in range(num_views)]
        label_y = np.squeeze(mat['Y'])
    elif dataset_name == "VGGFace":
        mat = sio.loadmat(os.path.join(data_path, 'VGGFace2_200_4Views.mat'))
        num_views = mat['X'].shape[1]
        X = mat['X']
        data_X = [X[0][i].astype('float32') for i in range(num_views)]
        label_y = np.squeeze(mat['Y'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if data_norm == 'standard':
        for i in range(len(data_X)):
            data_X[i] = skp.scale(data_X[i])
    elif data_norm == 'l2-norm':
        for i in range(len(data_X)):
            data_X[i] = data_X[i] / (np.linalg.norm(data_X[i], axis=1, keepdims=True) + 1e-8)
    elif data_norm == 'min-max':
        for i in range(len(data_X)):
            min_val = data_X[i].min(axis=0)
            max_val = data_X[i].max(axis=0)
            data_X[i] = (data_X[i] - min_val) / (max_val - min_val + 1e-8)

    if sample_size and sample_size < len(label_y):
        indices = np.random.choice(len(label_y), sample_size, replace=False)
        data_X = [X[indices] for X in data_X]
        label_y = label_y[indices]

    return data_X, label_y