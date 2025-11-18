import numpy as np


def GS_QR(A):
    """
    Classical Gram-Schmidt QR factorization.
    Returns Q (m x n) and R (n x n) such that A = QR.
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for k in range(n):
        v = A[:, k].copy()
        for j in range(k):
            R[j, k] = Q[:, j] @ v
            v = v - R[j, k] * Q[:, j]
        R[k, k] = np.linalg.norm(v)
        Q[:, k] = v / R[k, k]

    for k in range(n):
        if R[k, k] < 0:
            R[k, :] = -R[k, :]
            Q[:, k] = -Q[:, k]

    return Q, R


def HH_QR(A):
    """
    Householder QR factorization with implicit Q storage.
    """
    A_work = A.astype(float).copy()
    m, n = A_work.shape
    tau = np.zeros(n)

    for k in range(n):
        x = A_work[k:, k].copy()
        norm_x = np.linalg.norm(x)

        if norm_x < 1e-14:
            tau[k] = 0.0
            continue

        # Choose alpha with opposite sign of x[0] to avoid cancellation
        if x[0] >= 0:
            alpha = -norm_x
        else:
            alpha = norm_x

        v = x.copy()
        v[0] -= alpha

        # ✅ 正确的 tau 计算：tau = 2 * v[0]^2 / (v^T v)
        v_norm_sq = v @ v
        tau[k] = 2.0 * (v[0] ** 2) / v_norm_sq

        # 归一化 v 使得 v[0] = 1
        v = v / v[0]

        # 应用 Householder 变换
        A_block = A_work[k:, k:]
        A_work[k:, k:] = A_block - tau[k] * np.outer(v, v @ A_block)

        # 存储 v[1:]
        if k + 1 < m:
            A_work[k + 1:, k] = v[1:]

    R = np.triu(A_work[:n, :n])
    V = A_work

    return V, tau, R


def apply_Q(V, tau, x):
    """
    Compute Q @ x where Q is stored implicitly as Householder reflectors.
    Q = H_0 @ H_1 @ ... @ H_{n-1}

    Args:
        V: m x n matrix from HH_QR
        tau: length n array from HH_QR
        x: m-dimensional vector

    Returns:
        Q @ x
    """
    m, n = V.shape
    y = x.copy()

    for k in range(n - 1, -1, -1):
        v = np.zeros(m - k)
        v[0] = 1.0
        if k + 1 < m:
            v[1:] = V[k + 1:, k]

        y_sub = y[k:]
        y[k:] = y_sub - tau[k] * v * (v @ y_sub)

    return y


def apply_QT(V, tau, y):
    """
    Compute Q^T @ y where Q is stored implicitly as Householder reflectors.
    Q^T = H_{n-1} @ ... @ H_1 @ H_0

    Args:
        V: m x n matrix from HH_QR
        tau: length n array from HH_QR
        y: m-dimensional vector

    Returns:
        Q^T @ y
    """
    m, n = V.shape
    z = y.copy()

    for k in range(n):
        v = np.zeros(m - k)
        v[0] = 1.0
        if k + 1 < m:
            v[1:] = V[k + 1:, k]

        z_sub = z[k:]
        z[k:] = z_sub - tau[k] * v * (v @ z_sub)

    return z


def generate_R(n):
    """
    Generate random upper triangular matrix R with positive diagonal.

    Args:
        n: dimension of matrix

    Returns:
        R: n x n upper triangular matrix
    """
    R = np.triu(np.random.randn(n, n))

    for i in range(n):
        if R[i, i] < 0:
            R[i, :] = -R[i, :]

    return R


def generate_B(n):
    """
    Generate random matrix B with standard normal entries.

    Args:
        n: dimension of matrix

    Returns:
        B: n x n random matrix
    """
    B = np.random.randn(n, n)

    return B


if __name__ == "__main__":
    # Test the implementations
    np.random.seed(42)
    A = np.random.randn(6, 4)

    # Test GS_QR
    print("Testing GS_QR:")
    Q_gs, R_gs = GS_QR(A)
    print(f"  ||A - QR||_F = {np.linalg.norm(A - Q_gs @ R_gs, 'fro'):.2e}")
    print(f"  ||Q^T Q - I||_F = {np.linalg.norm(Q_gs.T @ Q_gs - np.eye(4), 'fro'):.2e}")
    print(f"  R diagonal positive? {np.all(np.diag(R_gs) > 0)}")

    # Test HH_QR
    print("\nTesting HH_QR:")
    V, tau, R_hh = HH_QR(A)

    # Reconstruct Q for verification
    Q_hh = np.eye(6)
    for k in range(3, -1, -1):
        v = np.zeros(6 - k)
        v[0] = 1.0
        if k + 1 < 6:
            v[1:] = V[k + 1:, k]
        Q_sub = Q_hh[k:, :]
        Q_hh[k:, :] = Q_sub - tau[k] * np.outer(v, v @ Q_sub)
    Q_hh = Q_hh[:, :4]

    print(f"  ||A - QR||_F = {np.linalg.norm(A - Q_hh @ R_hh, 'fro'):.2e}")
    print(f"  ||Q^T Q - I||_F = {np.linalg.norm(Q_hh.T @ Q_hh - np.eye(4), 'fro'):.2e}")

    # Test apply_Q and apply_QT
    print("\nTesting apply_Q and apply_QT:")
    x = np.random.randn(6)
    Qx = apply_Q(V, tau, x)
    print(f"  ||Q@x - apply_Q(V,tau,x)||_2 = {np.linalg.norm(Q_hh @ x - Qx):.2e}")

    y = np.random.randn(6)
    QTy = apply_QT(V, tau, y)
    print(f"  ||Q^T@y - apply_QT(V,tau,y)||_2 = {np.linalg.norm(Q_hh.T @ y - QTy):.2e}")