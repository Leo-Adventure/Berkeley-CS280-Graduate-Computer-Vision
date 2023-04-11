import numpy as np
def find_3d_points(P1, P2, matches):
    num_matches = len(matches)
    points_3d = np.zeros((num_matches, 3))
    err = 0

    for i, match in enumerate(matches):
        x1, y1, x2, y2 = match
        A = np.zeros((4, 4))
        A[0, :] = x1 * P1[2, :] - P1[0, :]
        A[1, :] = y1 * P1[2, :] - P1[1, :]
        A[2, :] = x2 * P2[2, :] - P2[0, :]
        A[3, :] = y2 * P2[2, :] - P2[1, :]

        _, _, V = np.linalg.svd(A)
        point_3d = V[-1, :3] / V[-1, 3]

        points_3d[i, :] = point_3d

        # Calculate error as the sum of the reprojection errors of the 3D points on both image planes
        proj1 = P1 @ np.append(point_3d, 1)
        proj1 = proj1[:2] / proj1[2]
        proj2 = P2 @ np.append(point_3d, 1)
        proj2 = proj2[:2] / proj2[2]

        err += np.sum((match[:2] - proj1)**2) + np.sum((match[2:] - proj2)**2)

    return points_3d, err / (2 * num_matches)
