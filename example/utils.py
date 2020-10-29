from typing import List
import numpy as np


def edit_dist(pred: List[int], label: List[int]) -> int:
    """Computes the edit distance between a predicted and label sequence."""
    # dists[i, j] = edit_dist(pred[:i], label[:i])
    pred_len, label_len = len(pred), len(label)
    dists = np.zeros((pred_len + 1, label_len + 1), dtype=int)

    dists[:, 0] = np.arange(pred_len + 1)
    dists[0, :] = np.arange(label_len + 1)

    for i, x in enumerate(pred):
        for j, y in enumerate(label):
            sub_delta = int(x != y)
            ins_delta = 1
            del_delta = 1

            substitution = dists[i, j] + sub_delta
            insertion = dists[i, j+1] + ins_delta  # pred[:i]  --> pred[:i+1]
            deletion = dists[i+1, j] + del_delta   # label[:j] --> label[:j+1]
            dists[i+1, j+1] = min(substitution, insertion, deletion)

    return dists[-1, -1].item()
