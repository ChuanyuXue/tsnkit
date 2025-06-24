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
3. Go `src` folder and run `python main.py --method=ALL --start=0 --end=38400`.

*Both `main` and `legacy` branches use the same logic (models & algorithms). However, we have refined the organization in the `main` branch by introducing a unified interface and standardized type notation to enhance maintainability and simplify the efforts to add new methods. The `legacy` branch houses the source code record used in the paper.*

> Note: As described in the paper, the benchmark results are generated on Chameleon Cloud with 8 nodes equipped with 2 AMD EPYC® CPUs, 64 cores per CPU. By setting the `timeout` parameter to 2 hours, it may take more than a week to complete 38400 * 17 experiments with the same environment. Therefore, attempting to replicate these results on a single local machine would be exceedingly time-intensive.

We provide the generated benchmark results in the `data` folder along with the visualization source code. By running the visualization code, you can reproduce the figures in the paper.

**Small scale benchmark demo:**

To quickly demonstrate the benchmark results, we provide a small scale benchmark with 256 experiments.

Run the following command to generate the benchmark results for all algorithms:

    ```
    python -m tsnkit.test.benchmark --methods ALL --ins 1-256
    ```
The estimated time to generate results for one algorithm, such as ls is 30 minutes. When running all the algorithms, results generation may take over 2 days. 

The default time limit for each test case for each algorithm is 10 minutes. For faster benchmark results, the time limit can be lowered:

    ```
    python -m tsnkit.test.benchmark --methods ALL --ins 1-256 -t 60
    ```

### Debug tool

The debug module can be used to run and validate methods across 48 test cases.
To validate individual modules, the following command can be used:
   
    ```
    python -m tsnkit.test.debug.at -t 60
    ```

In the above command, "at" represents one of the scheduling methods and be interchanged with other methods. 
This tool will take less than 1 hour to validate one method across all 48 cases.
To validate multiple modules, the following command can be used, with each desired method name separated by a space:

    ```
    python -m tsnkit.test.debug at cg cp_wa dt -t 60
    ```
Results for each algorithm are written to a file in the results directory of the debug folder containing information
on the runtime, memory usage, error messages (if any), and the simulation log for each test case. 
