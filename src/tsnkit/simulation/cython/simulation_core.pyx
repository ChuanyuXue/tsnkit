# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memset

ctypedef cnp.int32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def match_time_optimized(int t, list gcl):
    """
    Optimized Cython version of match_time function.
    
    This is a drop-in replacement for the Python match_time function
    that should provide better performance for the binary search.
    
    Args:
        t: Time value to match
        gcl: List of tuples [(start, end, queue), ...]
    
    Returns:
        int: Index of matching entry, or -1 if no match
    """
    if not gcl:
        return -1
        
    cdef int gcl_len = len(gcl)
    cdef int left = 0
    cdef int right = gcl_len - 1
    cdef int median
    cdef tuple right_entry = gcl[right]
    cdef tuple left_entry = gcl[0]
    
    # Fast access to start and end times
    cdef int right_start = right_entry[0]
    cdef int right_end = right_entry[1]
    cdef int left_start = left_entry[0]
    
    if right_start <= t < right_end:
        return right
    elif right_end <= t or t < left_start:
        return -1

    # Binary search with optimized tuple access
    cdef tuple left_tuple, median_tuple
    while True:
        median = (left + right) // 2
        if right - left <= 1:
            return left
        
        left_tuple = gcl[left]
        median_tuple = gcl[median]
        
        if left_tuple[0] <= t < median_tuple[0]:
            right = median
        else:
            left = median

 