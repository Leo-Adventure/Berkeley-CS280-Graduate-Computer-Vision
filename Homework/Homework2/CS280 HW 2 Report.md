# CS280 HW 2 Report

**Name: Mingqian Liao**

**SID: 3038745426**

**Date of Submission: 4/11 (using 5 slip days)**

<div style="page-break-after:always"></div>

## Q1 

### Step

1. Normalize the matches points
2. Construct matrix A using the coordinates of two corresponding points
3. Use SVD decomposition to calculate the solution of A, then select the eigenvector corresponding to the smallest eigenvalue to construct the fundamental matrix F
4. Do SVD decomposition to F, and to satisfy the contraint that the rank of fundamental is 2, we set the smallest singular value to 0, then recompute the F
5. Denormalization the fundamental matrix using $F = T_2^tFT_1$
6. Calculate the residual error between the input points and their corresponding epipolar lines using the computed fundamental matrix F.

The steps 4-6 are shown as below. 

```python
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
```

### Result

When it is 'house'

Residual in F = `0.008187572156680154`
The fundamental matrix is 

```matlab
[[-4.37863487e-08  4.46634445e-06 -2.51602232e-06]
 [ 1.00865294e-06 -3.84488799e-07  1.15175719e-02]
 [ 7.06333347e-05 -1.05420627e-02 -3.69044026e-01]]
```

When it is 'library'

Residual in F = `0.030859571129895978`
The fundamental matrix is  

```matlab
[[-3.66106207e-08 -4.73752936e-06  1.17665313e-03]
 [ 7.80422368e-07 -4.86861294e-08 -9.07364294e-03]
 [-1.20209710e-04  7.67259343e-03  1.83136614e-01]]
```

### Is this what you are directly optimizing using SVD when solving the homogeneous system? If yes, explain. If no, how does the objective relate to the residual?

No, it isn't. The residual is like a geometric error, but we use the sum of the squared errors between the epipolar constraint to optimize using SVD, which is the algebraic error. The residual can be considered as an indirect measure of the quality of the fundamental matrix F. A lower residual indicates that the computed F better satisfies the geometric constraints between the two images, meaning that the points and their corresponding epipolar lines are closer to each other. 

<div style="page-break-after:always"></div>

## Q2

When it is 'library'

All possible choices for rotation R are

```matlab
[[ 0.90726313,  0.01736635, -0.42020475],
       [ 0.0131854 , -0.99983046, -0.01285274],
       [-0.42035671,  0.00612025, -0.90733829]]
```

```matlab
[[ 0.9450887 , -0.02538812,  0.32582631],
       [ 0.02306004,  0.9996735 ,  0.01100601],
       [-0.32599935, -0.00288809,  0.94536558]]
```

All possible choices for translation t are

```matlab
[-0.92738597, -0.0181464 ,  0.37366559]
```

```matlab
[ 0.92738597,  0.0181464 , -0.37366559]
```

When it is 'house'

All possible choices for rotation R are

```matlab
[[ 0.99329437,  0.02994881, -0.11166633],
       [ 0.02876665, -0.99951191, -0.01218309],
       [-0.11197669,  0.00888913, -0.99367107]]
```

```matlab
[[ 0.98418285, -0.07054941,  0.16250196],
       [ 0.06907734,  0.99750302,  0.0146984 ],
       [-0.16313316, -0.00324071,  0.98659874]]
```

All possible choices for translation t are

```matlab
[-0.98927358, -0.04894846,  0.13762935]
```

```matlab
[ 0.98927358,  0.04894846, -0.13762935
```

<div style="page-break-after:always"></div>

## Q3

### Step

1. Iterate through each match (corresponding points in two images) and for each match, obtain the homogeneous coordinates `x1`, `y1`, `x2`, and `y2`
2. Create a 4x4 matrix A, which represents the system of linear equations derived from the given camera matrices P1 and P2 and the homogeneous coordinates of the matched points
3. Use SVD decomposition to compute the S, V, D matrices of the matrix A
4. To obtain the inhomogeneous 3D coordinates, divide the first three elements of the eigenvector by the last element.
5. Calculate the reprojection errors for both images by projecting the 3D point onto the image planes using the camera matrices P1 and P2. Subtract the original 2D points from the projected points, square the differences, and sum them up. Add this sum to the total reprojection error.

### Result

When it is 'library', the reconstruction error is `86.4976296661863`

When it is 'house', the reconstruction error is `142.23146459755145`

<div style="page-break-after:always"></div>

## Q4

When it is 'library'

![Figure_1](/Users/leo/Desktop/Figure_1.png)

When it is 'house'

![house](/Users/leo/Desktop/house.png)

