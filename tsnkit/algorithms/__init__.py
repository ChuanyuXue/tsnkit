"""
Author: Chuanyu (skewcy@gmail.com)
__init__.py (c) 2023
Desc: description
Created:  2023-10-09T00:02:25.471Z


DESC:
    This is the template for how to integrate 
    a new method into this benchmark.
    Please refer other methods for more details.
"""

from ..core import Statistics, Config, Result
from abc import ABC, abstractmethod

def benchmark(name: str,
              task_path: str,
              net_path: str,
              output_path: str = "./",
              workers: int = 1) -> Statistics:
    """ Implement this function to benchmark your method
    
    Args:
        name (str): experiment name
        task_path (str): file path of the task
        net_path (str): file path of the network
        output_path (str): output folder path
        workers (int): number of workers
    Returns:
        Statistics object
    """

    stat = Statistics(name)  ## Init empty stat
    try:
        ## ❕❕[NOTE]❕❕ Change _Method to your method class
        test = _Method(workers)  # type: ignore
        test.init(task_path, net_path)
        test.prepare()
        stat = test.solve()  ## Update stat
        if stat.result == Result.schedulable:
            test.output().to_csv(name, output_path)
        stat.content(name=name)
        return stat
    except KeyboardInterrupt:
        stat.content(name=name)
        return stat


class _Method(ABC):

    def __init__(self, name) -> None:
        self.name = name

    @abstractmethod
    def init(self, task_path: str, network_path: str) -> bool:
        """Init the method with the task and network

        Args:
            task_path (str): file path of the task
            network_path (str): file path of the network

        Returns:
            bool: True if success
        """
        pass

    @abstractmethod
    def prepare(self) -> bool:
        """Add constraints / preparation to the solver. Can simply return True if no constraints are needed (usually when no solver used).

        Returns:
            bool: True if success
        """
        pass

    @abstractmethod
    def solve(self) -> Statistics:
        """
        Solve the problem and return the statistics
        
        Returns:
            Statistics object
        """
        pass

    @abstractmethod
    def output(self) -> Config:
        """Return the standard output format

        Returns:
            Config object
        """
        pass
