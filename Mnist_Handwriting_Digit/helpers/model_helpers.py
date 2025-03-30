import numpy as np

def entropy_purity(S, y):
    """
    Hàm tính entropy purity của tập S với các lớp y
    :param
        - S: tập observations
        - y: tập labels
    
    :return
        - entropy: entropy của tập S
    """
    assert S.shape[0] == y.shape[0]

    N, d = S.shape
    unique_labels, counts = np.unique(y, return_counts = True)

    ps = [- (count / N) * np.log2(count / N) for count in counts]
    return np.sum(ps)


def information_gain(j, t, S, y):
    """
    Hàm tính information_gain sau khi chia cành
    :param
        - j: feature thứ j
        - t: ngưỡng chia
        - S: tập observations
        - y: tập labels
    :return
        - information_gain: information_gain sau khi chia cành
    """
    assert S.shape[0] == y.shape[0]
    N, d = S.shape
    
    Hs = entropy_purity(S, y)

    x_j = S[:, j]

    left_mask = x_j <= t
    right_mask = x_j > t

    left_observations, right_observations = S[left_mask], S[right_mask]
    left_labels, right_labels = y[left_mask], y[right_mask]

    p1 = left_observations.shape[0] / N
    p2 = right_observations.shape[0] / N
    H1 = entropy_purity(left_observations, left_labels)
    H2 = entropy_purity(right_observations, right_labels)

    return {
        "gain": Hs - (p1 * H1 + p2 * H2),
        "feature": j,
        "threshold": t,
        "left_observations": left_observations,
        "right_observations": right_observations,
        "left_labels": left_labels,
        "right_labels": right_labels,
    }



