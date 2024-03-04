import numpy as np
import scipy as sp
import tensorly as tl


def get_H(Us, n):  # U(-m).T
    # ab = Us[::-1]
    Hs = tl.tenalg.kronecker([u.T for u, i in zip(Us[::-1], reversed(range(len(Us)))) if i != n])
    return Hs

def update_cores( n, Us, Xs, cores,cores_pred,seq_length):

    T_hat = len(Xs)
    unfold_cores = get_unfold_tensor(cores, n)
    H = get_H(Us, n)  # U(-m).T
    for t in range(seq_length, len(cores_pred)+seq_length):

        unfold_Xs = get_unfold_tensor(Xs[t], n)
        # xx=np.dot(np.dot(Us[n].T, unfold_Xs), H.T)
        unfold_cores[t] = 1/2 * (np.dot(np.dot(Us[n].T, unfold_Xs), H.T) + cores_pred[t-seq_length])
    return unfold_cores


def get_cores( Xs, Us):
    cores = [tl.tenalg.multi_mode_dot(x, [u.T for u in Us], modes=[i for i in range(len(Us))]) for x in Xs]
    return cores

def initilizer(T_hat, Js, Rs, Xs):
    # initilize Us
    U = [np.random.random([j, r]) for j, r in zip(list(Js), Rs)]
    return U

def get_fold_tensor(tensor, mode, shape):
    if isinstance(tensor,list):
        return [ tl.base.fold(ten, mode, shape) for ten in tensor ]
    elif isinstance(tensor, np.ndarray):
        return tl.base.fold(tensor, mode, shape)
    else:
        raise TypeError(" 'tensor' need to be a list or numpy.ndarray")

def get_unfold_tensor(tensor, mode):  # 张量展开

    if isinstance(tensor, list):
        return [tl.base.unfold(ten, mode) for ten in tensor]
    elif isinstance(tensor, np.ndarray):
        return tl.base.unfold(tensor, mode)
    else:
        raise TypeError(" 'tensor' need to be a list or numpy.ndarray")

def update_Us( Us, Xs, unfold_cores, n,seq_length):

    T_hat = len(Xs)
    M = len(Us)
    begin_idx = seq_length

    H = get_H(Us, n)

    Bs = []
    for t in range(begin_idx, T_hat):
        unfold_X = get_unfold_tensor(Xs[t], n)
        Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
    b = np.sum(Bs, axis=0)
    # b = b.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
    U_, _, V_ = np.linalg.svd(b, full_matrices=False)
    Us[n] = np.dot(U_, V_)

    return Us


def MAE(y_pred, y_true):
    """
    计算 MAE（平均绝对误差）
    :param actual: 实际值列表
    :param predicted: 预测值列表
    :return: MAE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def RMSE(y_pred, y_true):
    """ RMSE """
    y_true, y_pred = np.array( y_true), np.array(y_pred)
    t1 = np.mean(( y_true - y_pred) **2)
    return np.sqrt(t1)
