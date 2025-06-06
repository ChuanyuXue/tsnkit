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

def match_time_wrapper(int t, cnp.ndarray[DTYPE_t, ndim=2] gcl):
    """Python wrapper for the fast match_time function"""
    return match_time_optimized(t, gcl.tolist())

@cython.boundscheck(False)
@cython.wraparound(False)
def process_qbv_fast(
    cnp.ndarray[DTYPE_t, ndim=1] links_src,
    cnp.ndarray[DTYPE_t, ndim=1] links_dst,
    cnp.ndarray[DTYPE_t, ndim=1] cycles,
    cnp.ndarray[DTYPE_t, ndim=1] available_times,
    cnp.ndarray[DTYPE_t, ndim=1] sizes,
    int t,
    dict gcl_dict,
    dict egress_q_dict,
    int T_PROC
):
    """
    Fast Cython implementation of QBV processing for all links at time t.
    Returns list of transmissions and updates to available_times.
    """
    cdef int num_links = links_src.shape[0]
    cdef int i, current_t, index, trans_delay, q, e
    cdef tuple link
    
    transmissions = []
    available_time_updates = {}
    
    for i in range(num_links):
        link = (links_src[i], links_dst[i])
        
        # Skip if not in GCL
        if link not in gcl_dict:
            continue
            
        current_t = t % cycles[i]
        gcl_list = gcl_dict[link]
        
        if len(gcl_list) == 0:
            continue
            
        gcl_array = np.array(gcl_list, dtype=np.int32)
        index = match_time_optimized(current_t, gcl_list)
        if index == -1:
            continue
            
        e = gcl_array[index][1]
        q = gcl_array[index][2]
        
        if t >= available_times[i] and len(egress_q_dict[link][q]) > 0:
            frame = egress_q_dict[link][q][0]
            trans_delay = sizes[frame[0]] * 8
            if e - current_t >= trans_delay:
                transmissions.append((link, frame, q))
                available_time_updates[link] = t + trans_delay
                
    return transmissions, available_time_updates

@cython.boundscheck(False)
@cython.wraparound(False)
def process_timer_fast(
    dict pool_dict,
    cnp.ndarray[DTYPE_t, ndim=1] src_nodes,
    cnp.ndarray[DTYPE_t, ndim=1] dst_nodes,
    int t,
    dict route_dict,
    dict queue_dict,
    dict egress_q_dict,
    int T_PROC
):
    """
    Fast Cython implementation of timer processing.
    """
    cdef int flow, ct
    cdef tuple link, frame, new_link
    
    new_pool = {}
    log_updates = {'send': [], 'receive': []}
    queue_updates = []
    
    # Convert to sets for faster lookup
    src_set = set(src_nodes)
    dst_set = set(dst_nodes)
    
    for link, vec in pool_dict.items():
        new_vec = []
        for ct, frame in vec:
            flow = frame[0]
            if t >= ct:
                # Check if this is a source node (talker)
                if link[0] in src_set:
                    log_updates['send'].append((flow, t))
                
                # Check if this is a destination node (listener)
                if link[1] in dst_set:
                    log_updates['receive'].append((flow, t - T_PROC))
                    continue
                
                # Forward to next hops
                if flow in route_dict and link[1] in route_dict[flow]:
                    for v in route_dict[flow][link[1]]:
                        new_link = (link[1], v)
                        queue_updates.append((new_link, queue_dict[frame][new_link], frame))
            else:
                new_vec.append((ct, frame))
        new_pool[link] = new_vec
    
    return new_pool, log_updates, queue_updates

@cython.boundscheck(False)
@cython.wraparound(False)
def release_tasks_fast(
    cnp.ndarray[DTYPE_t, ndim=1] periods,
    cnp.ndarray[DTYPE_t, ndim=1] instance_counts,
    cnp.ndarray[DTYPE_t, ndim=1] offset_maxs,
    int t,
    dict offset_dict,
    dict route_dict,
    dict src_dict,
    dict queue_dict
):
    """
    Fast Cython implementation of task release.
    """
    cdef int flow, period_val, offset_max_val, instance_count_val
    cdef tuple frame, link
    
    releases = []
    instance_updates = {}
    
    for flow in range(periods.shape[0]):
        period_val = periods[flow]
        instance_count_val = instance_counts[flow]
        offset_max_val = offset_maxs[flow]
        
        frame = (flow, instance_count_val % offset_max_val)
        
        if (t // period_val >= instance_count_val) and (t % period_val == offset_dict[frame]):
            src_node = src_dict[flow]
            if flow in route_dict and src_node in route_dict[flow]:
                for v in route_dict[flow][src_node]:
                    link = (src_node, v)
                    queue_id = queue_dict[frame][link]
                    releases.append((link, queue_id, frame))
            
            instance_updates[flow] = instance_count_val + 1
    
    return releases, instance_updates 