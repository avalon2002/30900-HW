import numpy as np
import matplotlib.pyplot as plt
from New_QR import GS_QR, HH_QR, generate_R, generate_B


def reconstruct_Q_from_HH(V, tau, n):
    """Reconstruct Q matrix from Householder representation."""
    m = V.shape[0]
    Q = np.eye(m)

    for k in range(n - 1, -1, -1):
        v = np.zeros(m - k)
        v[0] = 1.0
        if k + 1 < m:
            v[1:] = V[k + 1:, k]

        Q_sub = Q[k:, :]
        Q[k:, :] = Q_sub - tau[k] * np.outer(v, v @ Q_sub)

    return Q[:, :n]


def ensure_positive_diagonal(Q, R):
    """Ensure R has positive diagonal by adjusting both Q and R."""
    n = R.shape[1] if R.ndim > 1 else len(R)
    for i in range(n):
        if R[i, i] < 0:
            R[i, :] = -R[i, :]
            Q[:, i] = -Q[:, i]
    return Q, R


def compute_errors(Q_true, R_true, Q_hat, R_hat, A, n):
    """
    Compute the four relative errors.

    Returns:
        err_R: ||R - R_hat||_F / ||R||_F
        err_Q: ||Q - Q_hat||_F / ||Q||_F
        err_QR: ||A - Q_hat @ R_hat||_F / ||A||_F
        err_orth: ||I - Q_hat^T @ Q_hat||_F / ||I||_F
    """
    norm_R = np.linalg.norm(R_true, 'fro')
    norm_Q = np.linalg.norm(Q_true, 'fro')
    norm_A = np.linalg.norm(A, 'fro')
    norm_I = np.sqrt(n)

    err_R = np.linalg.norm(R_true - R_hat, 'fro') / norm_R
    err_Q = np.linalg.norm(Q_true - Q_hat, 'fro') / norm_Q
    err_QR = np.linalg.norm(A - Q_hat @ R_hat, 'fro') / norm_A

    I = np.eye(n)
    err_orth = np.linalg.norm(I - Q_hat.T @ Q_hat, 'fro') / norm_I

    return err_R, err_Q, err_QR, err_orth


def run_experiment():
    """Run the numerical experiment for various values of n."""
    n_values = [5, 10, 20, 50, 100, 200, 500]

    results = {
        'GS': {'err_R': [], 'err_Q': [], 'err_QR': [], 'err_orth': []},
        'HH': {'err_R': [], 'err_Q': [], 'err_QR': [], 'err_orth': []}
    }

    for n in n_values:
        print(f"Testing n = {n}...")

        # Reset seed for each n to get consistent test matrices
        np.random.seed(42)

        # Generate test matrices
        R_true = generate_R(n)
        B = generate_B(n)
        Q_true, _ = np.linalg.qr(B)
        A = Q_true @ R_true

        cond_A = np.linalg.cond(A)
        print(f"  cond(A) = {cond_A:.2e}")

        # Test Gram-Schmidt
        Q_gs, R_gs = GS_QR(A)
        Q_gs, R_gs = ensure_positive_diagonal(Q_gs, R_gs)

        err_R_gs, err_Q_gs, err_QR_gs, err_orth_gs = compute_errors(
            Q_true, R_true, Q_gs, R_gs, A, n
        )
        results['GS']['err_R'].append(err_R_gs)
        results['GS']['err_Q'].append(err_Q_gs)
        results['GS']['err_QR'].append(err_QR_gs)
        results['GS']['err_orth'].append(err_orth_gs)

        # Test Householder
        V, tau, R_hh = HH_QR(A)
        Q_hh = reconstruct_Q_from_HH(V, tau, n)
        Q_hh, R_hh = ensure_positive_diagonal(Q_hh, R_hh)

        err_R_hh, err_Q_hh, err_QR_hh, err_orth_hh = compute_errors(
            Q_true, R_true, Q_hh, R_hh, A, n
        )
        results['HH']['err_R'].append(err_R_hh)
        results['HH']['err_Q'].append(err_Q_hh)
        results['HH']['err_QR'].append(err_QR_hh)
        results['HH']['err_orth'].append(err_orth_hh)

    return n_values, results


def plot_results(n_values, results):
    """Plot the relative errors."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('QR Factorization: Relative Errors vs Matrix Size',
                 fontsize=16, fontweight='bold')

    error_types = [
        ('err_R', r'$\|R - \hat{R}\|_F / \|R\|_F$', 'Forward Error in R'),
        ('err_Q', r'$\|Q - \hat{Q}\|_F / \|Q\|_F$', 'Forward Error in Q'),
        ('err_QR', r'$\|A - \hat{Q}\hat{R}\|_F / \|A\|_F$',
         'Backward Error (Factorization)'),
        ('err_orth', r'$\|I - \hat{Q}^T\hat{Q}\|_F / \|I\|_F$',
         'Backward Error (Orthogonality)')
    ]

    for idx, (key, label, title) in enumerate(error_types):
        ax = axes[idx // 2, idx % 2]

        ax.loglog(n_values, results['GS'][key], 'o-', label='Gram-Schmidt',
                  linewidth=2, markersize=8, color='#e74c3c')
        ax.loglog(n_values, results['HH'][key], 's-', label='Householder',
                  linewidth=2, markersize=8, color='#3498db')

        # Add machine precision reference line
        ax.axhline(y=1e-15, color='gray', linestyle='--', linewidth=1,
                   alpha=0.5, label='Machine precision')

        ax.set_xlabel('Matrix size n', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('qr_errors.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_error_table(n_values, results):
    """Print errors in table format."""
    print("\n" + "=" * 100)
    print("GRAM-SCHMIDT QR")
    print("=" * 100)
    print(f"{'n':>6} {'err_R':>15} {'err_Q':>15} {'err_QR':>15} {'err_orth':>15}")
    print("-" * 100)
    for i, n in enumerate(n_values):
        print(f"{n:>6} {results['GS']['err_R'][i]:>15.6e} "
              f"{results['GS']['err_Q'][i]:>15.6e} "
              f"{results['GS']['err_QR'][i]:>15.6e} "
              f"{results['GS']['err_orth'][i]:>15.6e}")

    print("\n" + "=" * 100)
    print("HOUSEHOLDER QR")
    print("=" * 100)
    print(f"{'n':>6} {'err_R':>15} {'err_Q':>15} {'err_QR':>15} {'err_orth':>15}")
    print("-" * 100)
    for i, n in enumerate(n_values):
        print(f"{n:>6} {results['HH']['err_R'][i]:>15.6e} "
              f"{results['HH']['err_Q'][i]:>15.6e} "
              f"{results['HH']['err_QR'][i]:>15.6e} "
              f"{results['HH']['err_orth'][i]:>15.6e}")
    print("=" * 100)


if __name__ == "__main__":
    n_values, results = run_experiment()
    print_error_table(n_values, results)
    plot_results(n_values, results)