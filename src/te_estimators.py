import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors

def block_shuffle(series, block_size=5, seed=None):
    """
    Shuffle a time series in blocks (non-overlapping) to preserve short-range autocorrelation.
    """
    np.random.seed(seed)
    series = series.reset_index(drop=True)
    blocks = [series[i:i + block_size] for i in range(0, len(series), block_size)]
    np.random.shuffle(blocks)
    return pd.concat(blocks, ignore_index=True)


def discrete_te(source, target, nbins=10, pseudo_count=0.5):
    """
    Estimate transfer entropy from source X to target Y using a discrete binning approach.
    We assume a lag of 1, i.e. we estimate TE_{X→Y} = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t).
    """
    source = pd.Series(source).dropna().reset_index(drop=True)
    target = pd.Series(target).dropna().reset_index(drop=True)

    # Align series for lag = 1
    Y_next = target.iloc[1:].reset_index(drop=True)
    Y_t = target.iloc[:-1].reset_index(drop=True)
    X_t = source.iloc[:-1].reset_index(drop=True)

    Y_next_disc = pd.cut(Y_next, bins=nbins, labels=False)
    Y_t_disc = pd.cut(Y_t, bins=nbins, labels=False)
    X_t_disc = pd.cut(X_t, bins=nbins, labels=False)

    df = pd.DataFrame({'Y_next': Y_next_disc, 'Y_t': Y_t_disc, 'X_t': X_t_disc})

    joint_counts = df.groupby(['Y_next', 'Y_t', 'X_t']).size().reset_index(name='count')
    joint_counts['count'] += pseudo_count
    total_count = joint_counts['count'].sum()
    joint_counts['p_joint'] = joint_counts['count'] / total_count

    marginal_Y = df.groupby(['Y_next', 'Y_t']).size().reset_index(name='count')
    marginal_Y['count'] += pseudo_count * nbins
    marginal_Y['p_Y'] = marginal_Y['count'] / total_count

    marginal_YX = df.groupby(['Y_t', 'X_t']).size().reset_index(name='count')
    marginal_YX['count'] += pseudo_count * nbins
    marginal_YX['p_YX'] = marginal_YX['count'] / total_count

    joint_counts = joint_counts.merge(marginal_Y, on=['Y_next', 'Y_t'], how='left')
    joint_counts = joint_counts.merge(marginal_YX, on=['Y_t', 'X_t'], how='left')

    epsilon = 1e-10
    joint_counts['log_term'] = np.log2(
        (joint_counts['p_joint'] * joint_counts['p_Y'] + epsilon) /
        (joint_counts['p_Y'] * joint_counts['p_YX'] + epsilon)
    )

    te_estimate = np.sum(joint_counts['p_joint'] * joint_counts['log_term'])

    return te_estimate


def knn_te(source, target, k=5, exclude_temporal_neighbors=True):
    """
    Estimate transfer entropy from source X to target Y using a k-nearest neighbors (kNN) approach.
    We assume a lag of 1, i.e. we estimate TE_{X→Y} = I(X_t; Y_{t+1} | Y_t).

    Args:
        source, target: time series
        k: number of neighbors
        exclude_temporal_neighbors: whether to exclude neighbors within ±1 time step

    Returns:
        TE estimate in bits
    """
    s = pd.Series(source)
    t = pd.Series(target)

    df_aligned = pd.concat([s, t], axis=1).dropna().reset_index(drop=True)
    df_aligned.columns = ['source', 'target']

    if df_aligned.shape[0] < 2:
        return 0.0

    df_aligned = df_aligned.astype(float)

    Y_t = df_aligned['target'][:-1].values
    Y_next = df_aligned['target'][1:].values
    X_t = df_aligned['source'][:-1].values

    df_joint = pd.DataFrame({'Y_t': Y_t, 'X_t': X_t, 'Y_next': Y_next}).dropna().reset_index(drop=True)
    joint_data = df_joint.values

    if joint_data.shape[0] < k + 1:
        return 0.0

    marginal_Y = df_joint[['Y_t', 'Y_next']].values
    marginal_YX = df_joint[['Y_t', 'X_t']].values

    nbrs_joint = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(joint_data)
    distances_joint, _ = nbrs_joint.kneighbors(joint_data)
    epsilons = distances_joint[:, -1]
    n = joint_data.shape[0]

    # Optional: temporal exclusion helper
    def count_exclude_neighbors(data, epsilons):
        neighbors = []
        tree = NearestNeighbors(metric='euclidean').fit(data)
        for i in range(n):
            indices = tree.radius_neighbors([data[i]], radius=epsilons[i], return_distance=False)[0]
            valid = [j for j in indices if abs(j - i) > 1]
            neighbors.append(len(valid))
        return np.array(neighbors)

    if exclude_temporal_neighbors:
        n_Y = count_exclude_neighbors(marginal_Y, epsilons)
        n_YX = count_exclude_neighbors(marginal_YX, epsilons)
    else:
        nbrs_Y = NearestNeighbors(metric='euclidean').fit(marginal_Y)
        counts_Y = nbrs_Y.radius_neighbors(marginal_Y, radius=epsilons, return_distance=False)
        n_Y = np.array([len(neighbors) - 1 for neighbors in counts_Y])

        nbrs_YX = NearestNeighbors(metric='euclidean').fit(marginal_YX)
        counts_YX = nbrs_YX.radius_neighbors(marginal_YX, radius=epsilons, return_distance=False)
        n_YX = np.array([len(neighbors) - 1 for neighbors in counts_YX])

    psi_k = digamma(k)
    avg_psi_diff = np.mean(digamma(n_Y + 1) - digamma(n_YX + 1))
    te_estimate = psi_k - avg_psi_diff

    return te_estimate
