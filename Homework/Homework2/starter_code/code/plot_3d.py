import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d(points_3d, P1, P2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], marker='o', color='b', s=20)

    # Plot the camera centers
    camera_center1 = -np.linalg.inv(P1[:, :3]) @ P1[:, 3]
    camera_center2 = -np.linalg.inv(P2[:, :3]) @ P2[:, 3]
    ax.scatter(camera_center1[0], camera_center1[1], camera_center1[2], marker='^', color='r', s=100, label='Camera 1')
    ax.scatter(camera_center2[0], camera_center2[1], camera_center2[2], marker='^', color='g', s=100, label='Camera 2')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20, 20)

    ax.grid(True)

    plt.legend()
    plt.show()

