import numpy as np


def x_fall_within_sigma_1d(x, mean, cov, n_sigma=1):
    """
    Given a Gaussian distribution defined by mean, cov, determine if x falls within n_sigma.

    Args:
        x (float): the value to be tested
        mean (np.array): the mean of the Gaussian distribution
        cov (np.array): the covariance matrix of the Gaussian distribution
        n_sigma (int, optional): the number of sigma. Defaults to 1.

    Returns:
        bool: True if x falls within n_sigma, False otherwise.
    """
    assert cov.shape == (1, 1)
    assert mean.shape == (1,)
    # calculate the standard deviation
    std = np.sqrt(np.diag(cov))
    # calculate the lower and upper bound
    lower_bound = mean - n_sigma * std
    upper_bound = mean + n_sigma * std
    return lower_bound <= x <= upper_bound


def x_fall_within_sigma_2d(x, mean, cov, n_sigma=1):
    """Given a 2d Gaussian distribution defined by mean, cov, determine if
    a point x (x1, x2) falls within the ellipse defined by n_sigma.

    Args:
        x (np.array): the point to be tested
        mean (np.array): the mean of the Gaussian distribution
        cov (np.array): the covariance matrix of the Gaussian distribution
        n_sigma (int, optional): the number of sigma. Defaults to 1.

    Returns:
        bool: True if x falls within n_sigma, False otherwise.
    """
    assert cov.shape == (2, 2)
    assert mean.shape == (2,)

    eigv, eigw = np.linalg.eigh(cov)
    std = np.sqrt(eigv)  # major and minor radius of ellipse
    a = n_sigma * std[0]
    b = n_sigma * std[1]
    # apply eigenvector as transformation matrix
    x_trans = np.dot(eigw.T, x - mean)  # .T means backward rotation

    # apply ellipse equation
    return (((x_trans[0]) / a) ** 2 + ((x_trans[1]) / b) ** 2) <= 1


def x_fall_within_sigma_3d(x, mean, cov, n_sigma=1, verbose=False):
    """Given a 3d Gaussian distribution defined by mean, cov, determine if
    a point x (x1, x2, x3) falls within the ellipsoid defined by n_sigma.

    """
    assert cov.shape == (3, 3)
    assert mean.shape == (3,)
    eigv, eigw = np.linalg.eigh(cov)
    std = np.sqrt(eigv)  # radii of ellipsoid
    # calculate axis of ellpsoid
    a = n_sigma * std[0]
    b = n_sigma * std[1]
    c = n_sigma * std[2]

    # apply eigenvector as transformation matrix
    x_trans = np.dot(eigw.T, x - mean)  # .T means backward rotation

    # apply ellipsoid equation
    return (((x_trans[0]) / a) ** 2 + ((x_trans[1]) / b) ** 2 + ((x_trans[2]) / c) ** 2) <= 1


def _x_fall_within_sigma_2d_angle_method(x, mean, cov, n_sigma=1):
    """This function depicts how rotation happens, for illustration purpose only.
    Given a 2d Gaussian distribution defined by mean, cov, determine if
    a point x (x1, x2) falls within the ellipse defined by n_sigma.

    Args:
        x (np.array): the point to be tested
        mean (np.array): the mean of the Gaussian distribution
        cov (np.array): the covariance matrix of the Gaussian distribution
        n_sigma (int, optional): the number of sigma. Defaults to 1.

    Returns:
        bool: True if x falls within n_sigma, False otherwise.
    """
    assert cov.shape == (2, 2)
    assert mean.shape == (2,)

    eigv, eigw = np.linalg.eigh(cov)
    std = np.sqrt(
        eigv
    )  # major and minor radius of ellipse, not in any particular order
    a = n_sigma * std[0]
    b = n_sigma * std[1]
    # calculate the angle of the ellipse
    angle = np.arctan2(eigw[0][1], eigw[0][0])  # angle in radians, ccw
    # print(f"{a=}, {b=}, angle={180*angle/np.pi}")

    # translate X to origin and rotate cw
    x_trans = np.dot(
        x - mean,
        np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]),
    )  # cw rotation

    # apply ellipse equation
    return (((x_trans[0]) / a) ** 2 + ((x_trans[1]) / b) ** 2) <= 1


def _x_fall_within_sigma_3d_archive(x, mean, cov, n_sigma=1, verbose=False):
    """
    This function contains previously investigated rotation methods. They are incorrect.
    Given a 3d Gaussian distribution defined by mean, cov, determine if
    a point x (x1, x2, x3) falls within the ellipsoid defined by n_sigma.

    """
    assert cov.shape == (3, 3)
    assert mean.shape == (3,)
    eigv, eigw = np.linalg.eigh(cov)
    std = np.sqrt(eigv)  # radii of ellipsoid
    # calculate axis of ellpsoid
    a = n_sigma * std[0]
    b = n_sigma * std[1]
    c = n_sigma * std[2]

    # # use the quaternion method to calculate the angle
    # axis, angle = rotation_quaternion(eigw[:,0], np.array([0,0,1]))
    # if verbose:
    #     print(f"{eigv=}, \n{eigw=}, {eigw[:,0]=}, \n{axis=}, {np.degrees(angle)=}")
    # rotate and translate x back to origin
    # x_trans = rotate_vector(x - mean, axis, -angle)

    # axis, w = rotation_quaternion2(eigw[0], np.array([1,0,0]))
    # print(f"{axis=}, {w=}")
    # x_trans = rotate_vector2(x - mean, axis, w)

    # rotation using orthogonal Procrustes problem
    R = rotation_from_orthonormal(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), eigw)
    if verbose:
        print(f"{eigv=}, \neigw=\n{eigw}, \nR=\n{R}")
    x_trans = np.dot(eigw.T, x - mean)  # make sure it's the backward rotation

    # apply ellipse equation
    return (
        ((x_trans[0]) / a) ** 2 + ((x_trans[1]) / b) ** 2 + ((x_trans[2]) / c) ** 2
    ) <= 1


def rotation_from_orthonormal(A, B):
    """Calculate the rotation matrix that rotates A to B. The orthogonal Procrustes problem.
    Steps to calculate:
    1. Compute the cross-covariance matrix C between A and B.
    2. Perform Singular Value Decomposition (SVD) on C to obtain the rotation matrix R.
    3. Ensure that the determinant of R is positive to maintain the orientation.
    4. Check for a reflection in the rotation matrix and correct it if needed.
    """
    C = np.dot(B.T, A)
    U, _, Vt = np.linalg.svd(C)
    R = np.dot(Vt.T, U.T)

    # Ensure the determinant is positive
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = np.dot(Vt.T, U.T)

    return R


def rotation_quaternion(v, v_prime):
    """Calculate quaternion rotation. v and v_prime are vectors before and after rotation."""

    def normalize(v):
        return v / np.linalg.norm(v)

    axis = normalize(np.cross(v, v_prime))
    angle = np.arccos(
        np.clip(
            np.dot(v, v_prime) / (np.linalg.norm(v) * np.linalg.norm(v_prime)),
            -1.0,
            1.0,
        )
    )
    return axis, angle


def rotation_quaternion2(v, v_prime):
    """Calculate quaternion rotation. v and v_prime are vectors before and after rotation."""

    def normalize(v):
        return v / np.linalg.norm(v)

    # v_prime = normalize(v_prime)
    axis = normalize(np.cross(v, v_prime))
    w = np.sqrt((np.linalg.norm(v) ** 2) * (np.linalg.norm(v_prime) ** 2)) + np.dot(
        v, v_prime
    )
    return axis, w


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2],
        ]
    )


def rotate_vector(v, axis, angle):
    """Rotate vector v about axis by angle."""
    q_rot = np.array(
        [
            np.cos(angle / 2),
            axis[0] * np.sin(angle / 2),
            axis[1] * np.sin(angle / 2),
            axis[2] * np.sin(angle / 2),
        ]
    )
    R = quaternion_to_rotation_matrix(q_rot)
    v_rotated = np.dot(R, v)
    return v_rotated


def rotate_vector2(v, axis, w):
    """Rotate vector v about axis by angle."""
    q_rot = np.array([w, axis[0], axis[1], axis[2]])
    R = quaternion_to_rotation_matrix(q_rot)
    v_rotated = np.dot(R, v)
    return v_rotated
