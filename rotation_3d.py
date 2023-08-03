import numpy as np

def rotation_from_orthonormal(A, B):
    C = np.dot(B.T, A)
    U, _, Vt = np.linalg.svd(C)
    R = np.dot(Vt.T, U.T)

    # Ensure the determinant is positive
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = np.dot(Vt.T, U.T)

    return R

# Example usage:
A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Orthonormal set A
B = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # Orthonormal set B
B_cases = []
B_cases.append(B)
B_cases.append(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))
B_cases.append(np.array([[1, 0, 0], [0, np.sqrt(0.5), -np.sqrt(0.5)], [0, np.sqrt(0.5), np.sqrt(0.5)]]))

x = np.array([1, 0, 0])
for B in B_cases:
    print(f"{B=}")
    R = rotation_from_orthonormal(A, B)
    print(f"{x=}, {np.dot(R, x)=}")


def verify_rotation(A, B, R, tolerance=1e-6):
    A_rotated = np.dot(A, R)
    diff = np.abs(A_rotated - B)
    max_diff = np.max(diff)

    if max_diff <= tolerance:
        print("Verification passed. The rotation matrix R correctly aligns vectors from A to B.")
    else:
        print("Verification failed. The rotation matrix R does not correctly align vectors from A to B.")
        print(f"Maximum difference between rotated vectors and B: {max_diff}")

# Example usage:
A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Orthonormal set A
B = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # Orthonormal set B

R = rotation_from_orthonormal(A, B)

print("Rotation matrix from A to B:")
print(R)

# Verify the rotation
verify_rotation(A, B, R)
