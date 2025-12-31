
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------
# Sorting algorithms
# -----------------------------

def bubble_sort(array):
    start_time = time.perf_counter()
    n = len(array)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
                swapped = True
        if not swapped:
            break
    end_time = time.perf_counter() - start_time
    return array, end_time

def selection_sort(array):
    start_time = time.perf_counter()
    n = len(array)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if array[j] < array[min_idx]:
                min_idx = j
        array[i], array[min_idx] = array[min_idx], array[i]
    end_time = time.perf_counter() - start_time
    return array, end_time

def insertion_sort(array):
    start_time = time.perf_counter()
    for i in range(1, len(array)):
        key = array[i]
        j = i - 1
        while j >= 0 and key < array[j]:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = key
    end_time = time.perf_counter() - start_time
    return array, end_time

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result

def timed_merge_sort(arr):
    start = time.perf_counter()
    sorted_arr = merge_sort(arr)
    end_time = time.perf_counter() - start
    return sorted_arr, end_time

# -----------------------------
# Data generation
# -----------------------------

def data_generator():
    data = {}
    rng = np.random.default_rng(1234)
    sizes = [1000, 5000, 10000, 50000]
    for s in sizes:
        rand_arr = rng.integers(0, s, size=s)
        sorted_arr = np.sort(rand_arr)
        reverse_arr = sorted_arr[::-1]
        semi_arr = np.partition(rand_arr, s // 2)
        data[s] = {
            "random": rand_arr.tolist(),
            "sorted": sorted_arr.tolist(),
            "reverse": reverse_arr.tolist(),
            "semi": semi_arr.tolist(),
        }
    return data

# -----------------------------
# Benchmark runner
# -----------------------------

def benchmark_all():
    datasets = data_generator()
    results = {}
    for size, dists in datasets.items():
        results[size] = {}
        for dist_name, base_arr in dists.items():
            results[size][dist_name] = {}
            print(f'runing bubble for {dist_name} for {size}')
            _, tb = bubble_sort(base_arr.copy())
            print(f'runing selection for {dist_name} for {size}')
            _, ts = selection_sort(base_arr.copy())
            print(f'runing insertion for {dist_name} for {size}')
            _, ti = insertion_sort(base_arr.copy())
            print(f'runing merge for {dist_name} for {size}')
            _, tm = timed_merge_sort(base_arr.copy())
            results[size][dist_name] = {
                    'bubble': tb,
                    'selection': ts,
                    'insertion': ti,
                    'merge': tm
                }
    return results

# -----------------------------
# Tables and plotting
# -----------------------------

def results_to_wide_df(results):
    rows = []
    for size, dists in results.items():
        for dist, algos in dists.items():
            row = {'Size': size, 'Distribution': dist}
            row.update(algos)
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(['Size','Distribution']).reset_index(drop=True)
    return df

def plot_grouped_bars(df):
    """Create grouped bar charts: one figure per Size, distributions on x-axis, bars per algorithm."""
    algo_cols = [c for c in ['bubble','selection','insertion','merge'] if c in df.columns]
    colors = {
        'bubble':'#1f77b4',
        'selection':'#ff7f0e',
        'insertion':'#2ca02c',
        'merge':'#d62728'
    }

    for size in sorted(df['Size'].unique()):
        sub = df[df['Size']==size].copy()
        x_labels = list(sub['Distribution'])
        x = np.arange(len(x_labels))
        width = 0.18 if len(algo_cols)>0 else 0.6

        fig, ax = plt.subplots(figsize=(10, 5))
        for k, algo in enumerate(algo_cols):
            ax.bar(x + (k - (len(algo_cols)-1)/2)*width, sub[algo], width,
                   label=algo.capitalize(), color=colors.get(algo, None))

        ax.set_title(f"Sorting Performance — Size {size}")
        ax.set_xlabel("Distribution")
        ax.set_ylabel("Time (seconds)")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend(loc='upper left', ncol=len(algo_cols))
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        fig.tight_layout()
        out_path = f"sorting_performance_size_{size}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    # Summary chart: average across distributions per size
    agg = df.groupby(['Size']).mean(numeric_only=True)[algo_cols]
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(agg.index))
    width = 0.18 if len(algo_cols)>0 else 0.6
    for k, algo in enumerate(algo_cols):
        ax.bar(x + (k - (len(algo_cols)-1)/2)*width, agg[algo].values, width,
               label=algo.capitalize(), color=colors.get(algo, None))
    ax.set_title("Average Sorting Performance by Size (across distributions)")
    ax.set_xlabel("Size")
    ax.set_ylabel("Avg Time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in agg.index])
    ax.legend(loc='upper left', ncol=len(algo_cols))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig("sorting_performance_average_by_size.png", dpi=150)
    plt.close(fig)

# -----------------------------
# Run and save
# -----------------------------

if __name__ == "__main__":
    results = benchmark_all()
    df = results_to_wide_df(results)

    # Create figures
    plot_grouped_bars(df)

    # Print the table to console
    print(df.to_string(index=False))
