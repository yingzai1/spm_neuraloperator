import numpy as np
import sys

# The SciPy library is required for the statistical distribution test.
# You can install it with: pip install scipy
try:
    from scipy.stats import ks_2samp
except ImportError:
    print("Error: SciPy library not found. Please install it using 'pip install scipy'")
    sys.exit(1)


def print_array_stats(arr_name, arr, file_path):
    """Prints the shape, mean, and std of a given array."""
    print(f"\n[INFO] Array '{arr_name}' in file '{file_path}':")
    print(f"  - Shape: {arr.shape}")
    if np.issubdtype(arr.dtype, np.number):
        mean = arr.mean()
        std = arr.std()
        print(f"  - Stats: mean={mean:.6f}, std={std:.6f}")
    else:
        print("  - Stats: Not a numeric array.")


def statistical_comparison(arr1, arr2):
    """
    Performs a statistical comparison between two numerical arrays.
    Compares mean, std, and performs a Kolmogorov-Smirnov test.
    """
    # This comparison only makes sense for numeric data.
    if not np.issubdtype(arr1.dtype, np.number) or not np.issubdtype(
        arr2.dtype, np.number
    ):
        print("  - Content: Arrays are not numeric. Cannot perform statistical tests.")
        return

    print("  - Content: Arrays are not element-wise identical. Performing statistical tests...")

    # --- 1. Compare Mean and Standard Deviation ---
    max1, min1 = arr1.max(), arr1.min()
    max2, min2 = arr2.max(), arr2.min()
    mean1, std1 = arr1.mean(), arr1.std()
    mean2, std2 = arr2.mean(), arr2.std()

    print(f"    - Stats for file 1: mean={mean1:.6f}, std={std1:.6f}, min={min1:.6f}, max={max1:.6f}")
    print(f"    - Stats for file 2: mean={mean2:.6f}, std={std2:.6f}, min={min2:.6f}, max={max2:.6f}")

    # Use np.isclose for robust floating-point comparison
    if np.isclose(mean1, mean2) and np.isclose(std1, std2):
        print("    -> Verdict: Mean and Standard Deviation are nearly identical.")
    else:
        print("    -> Verdict: Mean and/or Standard Deviation are different.")

    # --- 2. Compare Distributions using Kolmogorov-Smirnov test ---
    # The K-S test works on 1-D arrays, so we flatten them first.
    flat_arr1 = arr1.flatten()
    flat_arr2 = arr2.flatten()

    # The null hypothesis of the K-S test is that the two samples are drawn from the same distribution.
    ks_statistic, p_value = ks_2samp(flat_arr1, flat_arr2)

    print(f"  - Distribution Test (Kolmogorov-Smirnov):")
    print(f"    - P-value: {p_value:.6f}")

    # We use a significance level (alpha) of 0.05.
    alpha = 0.05
    if p_value > alpha:
        print(f"    -> Verdict: The data likely comes from the SAME distribution (p > {alpha}).")
    else:
        print(f"    -> Verdict: The data likely comes from DIFFERENT distributions (p <= {alpha}).")


def compare_npz_files(file1_path, file2_path):
    """
    Loads two .npz files and compares their contents in detail.
    Includes statistical tests for non-identical numerical arrays.

    Args:
        file1_path (str): The path to the first .npz file.
        file2_path (str): The path to the second .npz file.

    Returns:
        bool: True if the files are identical, False otherwise.
    """
    try:
        # Load the two .npz files
        with np.load(file1_path) as data1, np.load(file2_path) as data2:
            files1 = set(data1.files)
            files2 = set(data2.files)
            all_keys = sorted(list(files1.union(files2)))
            overall_differences_found = False
            identical_arrays = []

            print("--- Comparing all arrays in both files ---")

            # --- 1. Iterate through each array and compare them ---
            for key in all_keys:
                arr1 = data1.get(key)
                arr2 = data2.get(key)

                # Case 1: Array only in file 1
                if arr1 is not None and arr2 is None:
                    print_array_stats(key, arr1, file1_path)
                    overall_differences_found = True
                    continue

                # Case 2: Array only in file 2
                if arr2 is not None and arr1 is None:
                    print_array_stats(key, arr2, file2_path)
                    overall_differences_found = True
                    continue

                # Case 3: Array in both files
                # Check for exact equality first, as it's the fastest.
                if np.array_equal(arr1, arr2):
                    identical_arrays.append(key)
                    continue

                # If not exactly equal, there is a difference.
                overall_differences_found = True
                print(f"\n[DIFFERENCE] Array '{key}':")

                # --- Compare data types (dtypes) ---
                if arr1.dtype != arr2.dtype:
                    print(f"  - Dtype mismatch: {arr1.dtype} vs {arr2.dtype}")

                # --- Compare shapes ---
                if arr1.shape != arr2.shape:
                    print("  - Shape mismatch detected.")
                    print_array_stats(key, arr1, file1_path)
                    print_array_stats(key, arr2, file2_path)
                    continue

                # --- Perform statistical comparison if shapes are the same ---
                statistical_comparison(arr1, arr2)

            # --- 2. Report identical arrays ---
            if identical_arrays:
                print(f"\n✅ IDENTICAL ARRAYS ({len(identical_arrays)} total):")
                for key in identical_arrays:
                    arr_shape = data1[key].shape
                    arr_dtype = data1[key].dtype
                    print(f"  - '{key}': shape={arr_shape}, dtype={arr_dtype}")

            print("-" * 40)
            if not overall_differences_found:
                print("\n✅ All arrays are identical. The files are the same.")
                return True
            else:
                print("\n❌ Found differences between the files as detailed above.")
                return False

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_dataset.py <file1.npz> <file2.npz>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    print(f"Comparing '{file1}' and '{file2}'...")
    print("-" * 40)

    compare_npz_files(file1, file2)

