import numpy as np
from sklearn.linear_model import RANSACRegressor

def normalize_points(matches):
    mean1 = np.mean(matches[:, :2], axis=0)
    mean2 = np.mean(matches[:, 2:], axis=0)

    dist1 = np.mean(np.sqrt(np.sum((matches[:, :2] - mean1)**2, axis=1)))
    dist2 = np.mean(np.sqrt(np.sum((matches[:, 2:] - mean2)**2, axis=1)))

    scale1 = np.sqrt(2) / dist1
    scale2 = np.sqrt(2) / dist2

    T1 = np.array([[scale1, 0, -scale1 * mean1[0]], [0, scale1, -scale1 * mean1[1]], [0, 0, 1]])
    T2 = np.array([[scale2, 0, -scale2 * mean2[0]], [0, scale2, -scale2 * mean2[1]], [0, 0, 1]])

    matches_norm = matches.copy()
    matches_norm[:, :2] = (matches[:, :2] - mean1) * scale1
    matches_norm[:, 2:] = (matches[:, 2:] - mean2) * scale2
    
    return matches_norm, T1, T2

def compute_fundamental_matrix(matches):
    A = np.zeros((len(matches), 9))
    for i, (p1, p2) in enumerate(zip(matches[:, :2], matches[:, 2:])):
        A[i] = [p1[0]*p2[0], p1[0]*p2[1], p1[0],
                p1[1]*p2[0], p1[1]*p2[1], p1[1],
                p2[0], p2[1], 1]

    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V

    return F

def fundamental_matrix(matches, ransac_threshold=1):
    matches_homogeneous = np.hstack([matches[:, :2], np.ones((len(matches), 1)),
                                      matches[:, 2:], np.ones((len(matches), 1))])

    # model = RANSACRegressor(residual_threshold=ransac_threshold)
    # model.fit(matches_homogeneous[:, :3], matches_homogeneous[:, 3:])
    # inliers = model.inlier_mask_
    
    matches_norm, T1, T2 = normalize_points(matches)
    F_norm = compute_fundamental_matrix(matches_norm)

    # Denormalize the fundamental matrix
    F = T2.T @ F_norm @ T1

    res_err = 0
    for i, (m1, m2) in enumerate(zip(matches[:, :2], matches[:, 2:])):
        m1 = np.append(m1, 1)
        m2 = np.append(m2, 1)
        res_err += (m1.T @ F @ m2) ** 2

    res_err /= len(matches)

    return F, res_err