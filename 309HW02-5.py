import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert, inv
import warnings

warnings.filterwarnings('ignore')


# Generate different types of matrices
def generate_matrix(matrix_type, n):
    """Generate a matrix of the specified type"""
    if matrix_type == 'randn':
        return np.random.randn(n, n)
    elif matrix_type == 'hilbert':
        return hilbert(n)
    elif matrix_type == 'pascal':
        from math import comb
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                P[i, j] = comb(i + j, i)
        return P
    elif matrix_type == 'magic':
        # Simple magic square generator (works well for odd n, approximate for even n)
        return generate_magic_square(n)


def generate_magic_square(n):
    """Generate a magic square"""
    magic = np.zeros((n, n), dtype=int)
    if n % 2 == 1:  # Odd order
        i, j = 0, n // 2
        for num in range(1, n * n + 1):
            magic[i, j] = num
            newi, newj = (i - 1) % n, (j + 1) % n
            if magic[newi, newj]:
                i = (i + 1) % n
            else:
                i, j = newi, newj
    else:  # Even order (simplified method)
        magic = np.arange(1, n * n + 1).reshape(n, n)
    return magic.astype(float)


# Compute different norms
def compute_norms(x, norms=['1', '2', 'inf']):
    """Compute various vector or matrix norms"""
    results = {}
    for norm_type in norms:
        if norm_type == '1':
            results['1'] = np.linalg.norm(x, ord=1)
        elif norm_type == '2':
            results['2'] = np.linalg.norm(x, ord=2)
        elif norm_type == 'inf':
            results['inf'] = np.linalg.norm(x, ord=np.inf)
    return results


# Analyze a single case
def analyze_case(A, x_true, b, norm_type):
    """Analyze the accuracy of a single linear system"""
    n = len(A)

    # Solve the linear system
    x_hat = np.linalg.solve(A, b)

    # Compute errors
    error = x_true - x_hat
    delta_b = A @ x_hat - b

    # Determine the norm type
    if norm_type == '1':
        ord_val = 1
    elif norm_type == '2':
        ord_val = 2
    else:  # 'inf'
        ord_val = np.inf

    norm_error = np.linalg.norm(error, ord=ord_val)
    norm_x = np.linalg.norm(x_true, ord=ord_val)
    norm_delta_b = np.linalg.norm(delta_b, ord=ord_val)
    norm_b = np.linalg.norm(b, ord=ord_val)

    # Compute condition number
    norm_A = np.linalg.norm(A, ord=ord_val)
    norm_A_inv = np.linalg.norm(inv(A), ord=ord_val)
    kappa = norm_A * norm_A_inv

    # Relative error
    rel_error = norm_error / norm_x if norm_x > 0 else 0
    bound = kappa * norm_delta_b / norm_b if norm_b > 0 else 0

    return rel_error, kappa, bound


# Detailed analysis for n = 5
def analyze_n5():
    """Detailed analysis for n = 5"""
    print("=" * 80)
    print("Detailed Analysis for n = 5")
    print("=" * 80)

    n = 5
    matrix_types = ['randn', 'hilbert', 'pascal', 'magic']
    norm_types = ['1', '2', 'inf']

    for mat_type in matrix_types:
        print(f"\nMatrix Type: {mat_type.upper()}")
        print("-" * 80)

        A = generate_matrix(mat_type, n)
        x_true = np.ones(n)
        b = A @ x_true

        print(f"Matrix A (first 3 rows):\n{A[:3, :]}\n")

        for norm_type in norm_types:
            rel_error, kappa, bound = analyze_case(A, x_true, b, norm_type)

            print(f"  {norm_type}-Norm:")
            print(f"    Relative Error ||x - x̂|| / ||x||:      {rel_error:.6e}")
            print(f"    Condition Number κ(A):                  {kappa:.6e}")
            print(f"    Error Bound κ(A) ||Δb|| / ||b||:       {bound:.6e}")
            print()


# Trend analysis for various n
def analyze_trends():
    """Analyze trends for different n"""
    print("\n" + "=" * 80)
    print("Trend Analysis (n = 5 to 100, step = 5)")
    print("=" * 80)

    n_values = range(5, 101, 5)
    matrix_types = ['randn', 'hilbert', 'pascal', 'magic']
    norm_types = ['1', '2', 'inf']

    # Store results
    results = {mat_type: {norm: {'rel_error': [], 'kappa': [], 'bound': []}
                          for norm in norm_types}
               for mat_type in matrix_types}

    for n in n_values:
        print(f"Processing n = {n}...")

        for mat_type in matrix_types:
            try:
                A = generate_matrix(mat_type, n)
                x_true = np.ones(n)
                b = A @ x_true

                for norm_type in norm_types:
                    rel_error, kappa, bound = analyze_case(A, x_true, b, norm_type)
                    results[mat_type][norm_type]['rel_error'].append(rel_error)
                    results[mat_type][norm_type]['kappa'].append(kappa)
                    results[mat_type][norm_type]['bound'].append(bound)
            except:
                # Fill with NaN if singular or computation fails
                for norm_type in norm_types:
                    results[mat_type][norm_type]['rel_error'].append(np.nan)
                    results[mat_type][norm_type]['kappa'].append(np.nan)
                    results[mat_type][norm_type]['bound'].append(np.nan)

    # Plot
    plot_results(n_values, results, matrix_types, norm_types)

    return results


def plot_results(n_values, results, matrix_types, norm_types):
    """Plot results"""
    n_list = list(n_values)

    for norm_type in norm_types:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{norm_type}-Norm Analysis', fontsize=16, fontweight='bold')

        for idx, mat_type in enumerate(matrix_types):
            ax = axes[idx // 2, idx % 2]

            rel_errors = results[mat_type][norm_type]['rel_error']
            kappas = results[mat_type][norm_type]['kappa']

            ax2 = ax.twinx()

            line1 = ax.semilogy(n_list, rel_errors, 'b-o', label='Relative Error', markersize=4)
            line2 = ax2.semilogy(n_list, kappas, 'r-s', label='Condition Number', markersize=4)

            ax.set_xlabel('n', fontsize=11)
            ax.set_ylabel('Relative Error', color='b', fontsize=11)
            ax2.set_ylabel('Condition Number κ(A)', color='r', fontsize=11)
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            ax.set_title(f'{mat_type.upper()} Matrix', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'analysis_{norm_type}_norm.png', dpi=150, bbox_inches='tight')
        print(f"Figure saved: analysis_{norm_type}_norm.png")
        plt.close()


# Compare direct solving vs using inverse
def compare_methods():
    """Compare direct solving and using matrix inverse"""
    print("\n" + "=" * 80)
    print("Comparison: Direct Solve vs Using Inverse")
    print("=" * 80)

    n_values = [5, 10, 20, 50]
    matrix_types = ['randn', 'hilbert', 'pascal']

    print(f"\n{'Matrix Type':<15} {'n':<5} {'Direct Solve Error':<20} {'Inverse Error':<20} {'Error Ratio':<15}")
    print("-" * 80)

    for mat_type in matrix_types:
        for n in n_values:
            try:
                A = generate_matrix(mat_type, n)
                x_true = np.ones(n)
                b = A @ x_true

                # Method 1: direct solve
                x_direct = np.linalg.solve(A, b)
                error_direct = np.linalg.norm(x_true - x_direct) / np.linalg.norm(x_true)

                # Method 2: using inverse
                A_inv = inv(A)
                x_inv = A_inv @ b
                error_inv = np.linalg.norm(x_true - x_inv) / np.linalg.norm(x_true)

                ratio = error_inv / error_direct if error_direct > 0 else np.inf

                print(f"{mat_type:<15} {n:<5} {error_direct:<20.6e} {error_inv:<20.6e} {ratio:<15.2f}")
            except:
                print(f"{mat_type:<15} {n:<5} {'Failed':<20} {'Failed':<20} {'-':<15}")


# Compute (1,1) element of A^(-1) without forming the full inverse
def compute_inv_11_element(A):
    """
    Compute the (1,1) element of A^(-1) without explicitly inverting A.
    Method: Solve Ax = e1, where e1 is the first standard basis vector.
    Then x[0] equals A^(-1)[0,0].
    """
    n = len(A)
    e1 = np.zeros(n)
    e1[0] = 1.0

    x = np.linalg.solve(A, e1)
    return x[0]


def test_inv_11():
    """Test computing the (1,1) element of the inverse"""
    print("\n" + "=" * 80)
    print("Computing the (1,1) Element of A^(-1)")
    print("=" * 80)

    n_values = [5, 10, 20]
    matrix_types = ['randn', 'hilbert', 'pascal']

    print(f"\n{'Matrix Type':<15} {'n':<5} {'Direct inv(A)[0,0]':<25} {'Indirect Method':<25} {'Difference':<15}")
    print("-" * 95)

    for mat_type in matrix_types:
        for n in n_values:
            try:
                A = generate_matrix(mat_type, n)

                A_inv = inv(A)
                direct = A_inv[0, 0]
                indirect = compute_inv_11_element(A)
                diff = abs(direct - indirect)

                print(f"{mat_type:<15} {n:<5} {direct:<25.10e} {indirect:<25.10e} {diff:<15.6e}")
            except:
                print(f"{mat_type:<15} {n:<5} {'Failed':<25} {'Failed':<25} {'-':<15}")


# Main program
def main():
    print("Numerical Analysis of Linear Systems")
    print("=" * 80)

    np.random.seed(42)

    analyze_n5()
    analyze_trends()
    compare_methods()
    test_inv_11()

    print("\n" + "=" * 80)
    print("Analysis Complete! Figures saved as PNG files.")
    print("=" * 80)


if __name__ == "__main__":
    main()
