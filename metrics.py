import torch
import random
import pickle
import numpy as np
import torchvision.transforms.functional as TF


def normalize(x, method="standard", axis=None):
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == "standard":
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(
                shape
            )
        elif method == "range":
            res = (x - np.min(y, axis=1).reshape(shape)) / (
                np.max(y, axis=1) - np.min(y, axis=1)
            ).reshape(shape)
        elif method == "sum":
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == "standard":
            res = (x - np.mean(x)) / np.std(x)
        elif method == "range":
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == "sum":
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def KLD(p, q):
    EPSILON = np.finfo("float").eps
    # EPSILON = np.finfo(np.float32).eps
    p = normalize(p, method="sum")
    q = normalize(q, method="sum")
    return np.sum(np.where(p != 0, p * np.log((p + EPSILON) / (q + EPSILON)), 0))


def AUC_Judd(saliency_map, fixation_map, jitter=False):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print("no fixation to predict")
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = TF.resize(
            saliency_map, fixation_map.shape, order=3, mode="constant"
        )
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method="range")

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds) + 2)
    fp = np.zeros(len(thresholds) + 2)
    tp[0] = 0
    tp[-1] = 1
    fp[0] = 0
    fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(
            S >= thresh
        )  # Total number of saliency map values above threshold
        tp[k + 1] = (k + 1) / float(
            n_fix
        )  # Ratio saliency map values at fixation locations above threshold
        fp[k + 1] = (above_th - k - 1) / float(
            n_pixels - n_fix
        )  # Ratio other saliency map values above threshold
    return np.trapz(tp, fp)  # y, x


def NSS(saliency_map, fixation_map):
    s_map = np.array(saliency_map, copy=False)
    f_map = np.array(fixation_map, copy=False) > 0.5
    if s_map.shape != f_map.shape:
        s_map = TF.resize(s_map, f_map.shape)
    # Normalize saliency map to have zero mean and unit std
    s_map = normalize(s_map, method="standard")
    # Mean saliency value at fixation locations
    return np.mean(s_map[f_map])


def CC(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = TF.resize(
            map1, map2.shape, order=3, mode="constant"
        )  # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have zero mean and unit std
    map1 = normalize(map1, method="standard")
    map2 = normalize(map2, method="standard")
    # Compute correlation coefficient
    return np.corrcoef(map1.ravel(), map2.ravel())[0, 1]


def SIM(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = TF.resize(
            map1, map2.shape, order=3, mode="constant"
        )  # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have values between [0,1] and sum up to 1
    map1 = normalize(map1, method="range")
    map2 = normalize(map2, method="range")
    map1 = normalize(map1, method="sum")
    map2 = normalize(map2, method="sum")
    # Compute histogram intersection
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)


def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_value(data, key, nested_key):

    key_str = key
    nested_key_str = f"{nested_key:04}"  # Format nested_key as a 4-digit string

    if key_str in data:
        if nested_key_str in data[key_str]:
            return data[key_str][nested_key_str]
        else:
            return f"Nested key {nested_key_str} does not exist in key {key_str}."
    else:
        return f"Key {key_str} does not exist in the data."


def gen_binary_map(fixations1, saliency, width, height, dataset_name):
    fixations1 = np.array(fixations1, dtype=np.float32).copy()

    if dataset_name == "eye":
        # Flip Y-axis
        fixations1[:, 1] = 1 - fixations1[:, 1]

    elif dataset_name == "pvs":
        # Flip X-axis and Y-axis
        fixations1[:, 0] = 1 - fixations1[:, 0]
        fixations1[:, 1] = 1 - fixations1[:, 1]

    elif dataset_name == "360":
        # No flipping
        pass

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Expected '360', 'eye', or 'pvs'."
        )

    fixations1 = fixations1 * np.array([width, height]) - [1, 1]
    fixations1 = fixations1.astype(int)

    fixmap1 = np.zeros(saliency.shape, dtype=np.float32)

    for iFix in range(fixations1.shape[0]):
        fixmap1[
            fixations1[iFix, 1],
            fixations1[iFix, 0]
        ] += 1

    return fixmap1


def evaluation_metrics(data, s_map, gt, vid, fid, dataset_name):

    height = 224
    width = 384

    tensor_shape = s_map.shape
    batch_size = tensor_shape[0]

    g_cc = []
    g_nss = []
    g_kl = []
    g_auc_judd = []
    g_sim = []

    EPSILON = np.finfo("float").eps
    VerticalWeighting = np.sin(
        np.linspace(0, np.pi, height)
    )  # latitude weighting 
    
    for i in range(batch_size):
        saliency = s_map[i].squeeze().cpu().numpy().astype(np.float32)
        groundTH = gt[i].squeeze().cpu().numpy().astype(np.float32)

        fixation = get_value(data, vid[i], int(fid[i]))

        fixation_map = gen_binary_map(fixation, saliency, width, height, dataset_name)

        saliency = saliency * VerticalWeighting[:, None] + EPSILON
        groundTH = groundTH * VerticalWeighting[:, None] + EPSILON
        
        cc_score = CC(saliency, groundTH)
        g_cc.append(cc_score)

        nss_score = NSS(saliency, fixation_map)
        g_nss.append(nss_score)

        kldiv_score = (KLD(groundTH, saliency) + KLD(saliency, groundTH)) / 2
        g_kl.append(kldiv_score)

        auc_judd_score = AUC_Judd(saliency, fixation_map)
        g_auc_judd.append(auc_judd_score)

        sim_score = SIM(saliency, groundTH)
        g_sim.append(sim_score)

    return g_cc, g_nss, g_kl, g_auc_judd, g_sim

