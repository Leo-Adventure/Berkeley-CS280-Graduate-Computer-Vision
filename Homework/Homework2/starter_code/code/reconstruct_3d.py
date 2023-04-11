import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from fundamental_matrix import fundamental_matrix
from find_rotation_translation import find_rotation_translation
from find_3d_points import find_3d_points
from plot_3d import plot_3d

def load_matrix_from_mat_file(file_path):
    mat_data = loadmat(file_path)
    for key in mat_data:
        if not key.startswith("__"):
            return mat_data[key]

def reconstruct_3d(name):
    # Load images, K matrices and matches
    data_dir = os.path.join('../data', name)

    # Images
    I1 = cv2.imread(os.path.join(data_dir, f'{name}1.jpg'))
    I2 = cv2.imread(os.path.join(data_dir, f'{name}2.jpg'))

    # K matrices

    K1_file = os.path.join(data_dir, f'{name}1_K.mat')
    K2_file = os.path.join(data_dir, f'{name}2_K.mat')

    K1 = load_matrix_from_mat_file(K1_file)
    K2 = load_matrix_from_mat_file(K2_file)
   

    # Corresponding points
    matches = np.loadtxt(os.path.join(data_dir, f'{name}_matches.txt'))

    # Visualize matches
    if True:
        plt.figure()
        plt.imshow(np.concatenate((I1, I2), axis=1))
        plt.scatter(matches[:, 0], matches[:, 1], marker='+', color='r')
        plt.scatter(matches[:, 2] + I1.shape[1], matches[:, 3], marker='+', color='r')
        for i in range(matches.shape[0]):
            plt.plot([matches[i, 0], matches[i, 2] + I1.shape[1]], [matches[i, 1], matches[i, 3]], 'r')
        plt.show()

    # Find fundamental matrix
    F, res_err = fundamental_matrix(matches)  # You write this one!
    print('Residual in F =', res_err)
    print('The fundamental matrix is ', F)

    E = K2.T @ F @ K1  # The essential matrix

    # Rotation and translation of camera 2
    R, t = find_rotation_translation(E)  # You write this one!
    print("All possible choices of translation t is ", t, ", and all possible choices of rotation R is", R)

    P1 = np.hstack([K1, np.zeros((3, 1))])

    num_points = np.zeros((len(t), len(R)))
    errs = np.inf * np.ones((len(t), len(R)))

    for ti, t2 in enumerate(t):
        for ri, R2 in enumerate(R):
            P2 = K2 @ np.hstack([R2, t2.reshape(-1, 1)])

            points_3d, err = find_3d_points(P1=P1, P2=P2, matches=matches)
            errs[ti, ri] = err

            Z1 = points_3d[:, 2]
            Z2 = R2[2, :] @ points_3d.T + t2[2]
            num_points[ti, ri] = np.sum((Z1 > 0) & (Z2 > 0))

    ti, ri = np.unravel_index(np.argmax(num_points), num_points.shape)

    print(f'Reconstruction error = {errs[ti, ri]}')

    t2 = t[ti]
    R2 = R[ri]
    P2 = K2 @ np.hstack([R2, t2.reshape(-1, 1)])

    points_3d, _ = find_3d_points(P1, P2, matches)

    plot_3d(points_3d, P1, P2)




reconstruct_3d('house')  # or 'library'