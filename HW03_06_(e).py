import numpy as np
from New_QR import GS_QR, HH_QR, apply_QT

C14_TRUE = 2006.787453080206


def build_vandermonde_and_b(m=100, n=15):
    """
    Construct Vandermonde matrix A and vector b:
        alpha_i = i/(m-1)
        A_{i,j} = alpha_i^j, j=0,...,n-1
        b_i = exp(sin(4 * alpha_i))
    """
    i = np.arange(m)
    alpha = i / (m - 1)
    A = np.vander(alpha, N=n, increasing=True)
    b = np.exp(np.sin(4 * alpha))
    return A, b


def method_i_qr_on_A_GS(A, b):
    """
    (i) QR on A using Gram-Schmidt, solve R x = Q^T b
    """
    Q, R = GS_QR(A)
    y = Q.T @ b
    x = np.linalg.solve(R, y)
    return x


def method_ii_qr_on_augmented_GS(A, b):
    """
    (ii) QR on [A, b] using Gram-Schmidt
    """
    m, n = A.shape
    A_aug = np.column_stack((A, b))
    Q_aug, R_aug = GS_QR(A_aug)

    R = R_aug[:n, :n]
    c_vec = R_aug[:n, -1]
    x = np.linalg.solve(R, c_vec)
    return x


def method_i_qr_on_A_HH(A, b):
    """
    (i) QR on A using Householder, solve R x = Q^T b
    """
    V, tau, R = HH_QR(A)
    y = apply_QT(V, tau, b)
    x = np.linalg.solve(R, y[:len(R)])
    return x


def method_ii_qr_on_augmented_HH(A, b):
    """
    (ii) QR on [A, b] using Householder
    """
    m, n = A.shape
    A_aug = np.column_stack((A, b))
    V_aug, tau_aug, R_aug = HH_QR(A_aug)

    R = R_aug[:n, :n]
    c_vec = R_aug[:n, -1]
    x = np.linalg.solve(R, c_vec)
    return x


def method_iii_normal_equations(A, b):
    """
    (iii) Solve normal equations A^T A x = A^T b
    """
    ATA = A.T @ A
    ATb = A.T @ b
    x = np.linalg.solve(ATA, ATb)
    return x


def main():
    m = 100
    n = 15
    A, b = build_vandermonde_and_b(m, n)

    print("=" * 80)
    print(f"Vandermonde Least Squares Problem: m={m}, n={n}")
    print(f"True value of c_14: {C14_TRUE:.15f}")
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    print("=" * 80)

    # Gram-Schmidt methods
    print("\nGRAM-SCHMIDT QR:")
    print("-" * 80)

    x_gs_i = method_i_qr_on_A_GS(A, b)
    c14_gs_i = x_gs_i[14]
    err_gs_i_abs = abs(c14_gs_i - C14_TRUE)
    err_gs_i_rel = err_gs_i_abs / abs(C14_TRUE)

    print(f"(i)  QR of A:")
    print(f"     c_14 = {c14_gs_i:.15f}")
    print(f"     Absolute error: {err_gs_i_abs:.6e}")
    print(f"     Relative error: {err_gs_i_rel:.6e}")

    x_gs_ii = method_ii_qr_on_augmented_GS(A, b)
    c14_gs_ii = x_gs_ii[14]
    err_gs_ii_abs = abs(c14_gs_ii - C14_TRUE)
    err_gs_ii_rel = err_gs_ii_abs / abs(C14_TRUE)

    print(f"\n(ii) QR of [A, b]:")
    print(f"     c_14 = {c14_gs_ii:.15f}")
    print(f"     Absolute error: {err_gs_ii_abs:.6e}")
    print(f"     Relative error: {err_gs_ii_rel:.6e}")

    # Householder methods
    print("\n" + "=" * 80)
    print("HOUSEHOLDER QR:")
    print("-" * 80)

    x_hh_i = method_i_qr_on_A_HH(A, b)
    c14_hh_i = x_hh_i[14]
    err_hh_i_abs = abs(c14_hh_i - C14_TRUE)
    err_hh_i_rel = err_hh_i_abs / abs(C14_TRUE)

    print(f"(i)  QR of A:")
    print(f"     c_14 = {c14_hh_i:.15f}")
    print(f"     Absolute error: {err_hh_i_abs:.6e}")
    print(f"     Relative error: {err_hh_i_rel:.6e}")

    x_hh_ii = method_ii_qr_on_augmented_HH(A, b)
    c14_hh_ii = x_hh_ii[14]
    err_hh_ii_abs = abs(c14_hh_ii - C14_TRUE)
    err_hh_ii_rel = err_hh_ii_abs / abs(C14_TRUE)

    print(f"\n(ii) QR of [A, b]:")
    print(f"     c_14 = {c14_hh_ii:.15f}")
    print(f"     Absolute error: {err_hh_ii_abs:.6e}")
    print(f"     Relative error: {err_hh_ii_rel:.6e}")

    # Normal equations
    print("\n" + "=" * 80)
    print("NORMAL EQUATIONS:")
    print("-" * 80)

    x_normal = method_iii_normal_equations(A, b)
    c14_normal = x_normal[14]
    err_normal_abs = abs(c14_normal - C14_TRUE)
    err_normal_rel = err_normal_abs / abs(C14_TRUE)

    print(f"(iii) A^T A x = A^T b:")
    print(f"      c_14 = {c14_normal:.15f}")
    print(f"      Absolute error: {err_normal_abs:.6e}")
    print(f"      Relative error: {err_normal_rel:.6e}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("-" * 80)
    print(f"{'Method':<30} {'c_14':>20} {'Rel. Error':>15}")
    print("-" * 80)
    print(f"{'GS QR of A':<30} {c14_gs_i:>20.10f} {err_gs_i_rel:>15.6e}")
    print(f"{'GS QR of [A,b]':<30} {c14_gs_ii:>20.10f} {err_gs_ii_rel:>15.6e}")
    print(f"{'HH QR of A':<30} {c14_hh_i:>20.10f} {err_hh_i_rel:>15.6e}")
    print(f"{'HH QR of [A,b]':<30} {c14_hh_ii:>20.10f} {err_hh_ii_rel:>15.6e}")
    print(f"{'Normal equations':<30} {c14_normal:>20.10f} {err_normal_rel:>15.6e}")
    print("=" * 80)


if __name__ == "__main__":
    main()