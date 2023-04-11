import numpy as np

def find_rotation_translation(E):
    # Perform a singular value decomposition (SVD) of E
    U, S, V = np.linalg.svd(E)

    # Define the possible rotation matrices and translation vectors
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    R1 = U @ W @ V
    R2 = U @ W.T @ V

    t1 = U[:, 2]
    t2 = -U[:, 2]

    possible_rotations = [R1,R2]
    possible_translations = [t1, t2]

    return possible_rotations, possible_translations
