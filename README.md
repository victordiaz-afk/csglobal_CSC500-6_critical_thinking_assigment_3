# csglobal_CSC500-6_critical_thinking_assigment_3

# Sorting Algorithm Benchmark

This project benchmarks four sorting algorithms—**Bubble Sort**, **Selection Sort**, **Insertion Sort**, and **Merge Sort**—across different dataset sizes and distributions. It measures execution time and generates grouped bar charts for visual comparison.

##  Features

- Deterministic data generation (fixed RNG seed)
- Benchmarks across multiple sizes: `1000`, `5000`, `10000`, `50000`
- Data distributions: **random**, **sorted**, **reverse**, **semi** (partially ordered via `np.partition`)
- Outputs:
 
  - Charts: `sorting_performance_size_<SIZE>.png`
  - Summary chart: `sorting_performance_average_by_size.png`


