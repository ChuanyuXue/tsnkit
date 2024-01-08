<!--
Author: <Chuanyu> (skewcy@gmail.com)
benchmark.md (c) 2023
Desc: description
Created:  2023-11-28T21:00:28.491Z
-->


# Benchmark generation

Following are the steps to generate the benchmark results used in the paper.

**Reproducing benchmark paper results:**

1. Check out to `legacy` branch.
2. Download `data.gz` from git-lfs, and unzip it to `data` folder. (Or generate it using `data/input/generate_data.ipynb`)
3. Go `src` foder and run `python main.py --method=ALL --start=0 --end=38400`.

*Both `main` and `legacy` branches use the same logic (models & algorithms). However, we have refined the organization in the `main` branch by introducing a unified interface and standardized type notation to enhance maintainability and simplify the efforts to add new methods. The `legacy` branch houses the source code record used in the paper.*

Please feel free to contact us if you have any obstacle in reproducing our result.