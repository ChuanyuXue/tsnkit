<!--
Author: <Chuanyu> (skewcy@gmail.com)
benchmark.md (c) 2023
Desc: description
Created:  2023-11-28T21:00:28.491Z
-->


# Benchmark generation

Following are the steps to generate the benchmark results used in the paper.

### Reproducing complete benchmark paper results:

**Run simulation:**


1. Check out to `legacy` branch.
2. Download `data.gz` from git-lfs, and unzip it to `data` folder. (Or generate it using `data/input/generate_data.ipynb`)
3. Go `src` foder and run `python main.py --method=ALL --start=0 --end=38400`.

*Both `main` and `legacy` branches use the same logic (models & algorithms). However, we have refined the organization in the `main` branch by introducing a unified interface and standardized type notation to enhance maintainability and simplify the efforts to add new methods. The `legacy` branch houses the source code record used in the paper.*

> Note: As described in the paper, the benchmark results are generated on Chameleon Cloud with 8 nodes equipped with 2 AMD EPYC® CPUs, 64 cores per CPU. By setting the `timeout` parameter to 2 hours, it may take more than a week to complete 38400 * 17 experiments with the same environment. Therefore, attempting to replicate these results on a single local machine would be exceedingly time-intensive.

We provide the generated benchmark results in the `data` folder along with the visualization source code. By running the visualization code, you can reproduce the figures in the paper.

**Small scale benchmark demo:**

To quickly demonstrate the benchmark results, we provide a small scale benchmark with 256 experiments. Follow the steps below to produce the results:

1. Check out to `main` branch and run following command to generate the problem instances:

   ```
   python -m tsnkit.data.generator --num_ins 1 --num_stream 10,40,70,100,130,160,190,220 --num_sw 8,18,28,38,48,58,68,78 --period 3, 4 --size 2 --deadline 1 --topo 0,1 --output "../data/"
   ``` 

2. Check out to `legacy` branch and run the following command to generate the benchmark results:

    ```
    python main.py --method=ALL --start=0 --end=256
    ```



