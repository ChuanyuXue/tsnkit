.. tsnkit documentation master file, created by
   sphinx-quickstart on Tue Nov 28 13:37:20 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TSNKit 
==================================
TSNKit is an open-source toolkit for configuring Time-Sensitive Networking (TSN) networks developed in Python. It is designed to be used by network operators, researchers, and students to reproduce the classical TSN scheduling algorithms and rapidly configure the TSN network.  

If you use this software in a scholarly publication, please cite the following paper. The detailed system model, algorithms, and benchmarking results can also be found in the `paper <https://arxiv.org/abs/2305.16772>`_.

.. code-block:: 

    @article{xue2023real,
      title={Real-Time Scheduling for Time-Sensitive Networking: A Systematic Review and Experimental Study},
      author={Xue, Chuanyu and Zhang, Tianyu and Zhou, Yuanbin and Han, Song},
      journal={arXiv preprint arXiv:2305.16772},
      year={2023}
    }

Documentation
--------------

.. toctree::
   :maxdepth: 1

   quickstarted.md
   dataprep.md
   schedule.md
   simulation.md
   testbed.md
   benchmark.md
   

Python APIs
------------

.. toctree::
   :maxdepth: 1

   APIs/modules


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
