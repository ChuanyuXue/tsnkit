#!/usr/bin/env python3
"""
Animate TSN packet movements with Straight Orthogonal Edges.

Changes:
- Layout: Automatically rotates the graph to align with X/Y axes (fixing "diamond" rotations).
- Routing: Uses straight lines (no L-bends).
- Constraint: Edges will be vertical/horizontal IF the topology allows (e.g., Grid/Mesh).
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import pandas as pd


def parse_link(s: str) -> Tuple[int, int]:
    s = s.strip().strip('"').strip()
    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError(f"Bad link format: {s}")
    u_str, v_str = s[1:-1].split(",")
    return int(u_str.strip()), int(v_str.strip())


def load_topology(path: str) -> nx.DiGraph:
    df = pd.read_csv(path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        u, v = parse_link(row["link"])
        G.add_edge(
            u,
            v,
            q_num=int(row.get("q_num", 0)),
            rate=float(row.get("rate", 0)),
            t_proc=float(row.get("t_proc", 0)),
            t_prop=float(row.get("t_prop", 0)),
        )
    return G


def load_tasks(path: str) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    df = pd.read_csv(path)
    flow_to_src: Dict[int, int] = {}
    flow_to_dsts: Dict[int, List[int]] = {}
    for _, row in df.iterrows():
        fid = int(row["stream"]) if "stream" in row else int(row["stream"])
        src = int(row["src"])
        dst_list = row["dst"]
        try:
            dsts = list(map(int, ast.literal_eval(dst_list)))
        except Exception:
            dsts = [int(dst_list)]
        flow_to_src[fid] = src
        flow_to_dsts[fid] = dsts
    return flow_to_src, flow_to_dsts


@dataclass
class Segment:
    flow: int
    u: int
    v: int
    t0_ns: int
    t1_ns: int

    def t0_anim(self, ns_per_sec: float) -> float:
        return self.t0_ns / ns_per_sec

    def t1_anim(self, ns_per_sec: float) -> float:
        return self.t1_ns / ns_per_sec


def build_segments(log_path: str) -> List[Segment]:
    df = pd.read_csv(log_path)
    df["stream"] = df["stream"].astype(int)
    df["di"] = df["di"].astype(int)
    df["time"] = df["time"].astype(int)
    
    # We still need u_v from the log for the Send (di=1) events
    df["u_v"] = df["link"].map(parse_link)
    
    # Sort strictly by time, then di
    # We want to process 'di=1' (send) before 'di=0' (receive) if times are equal 
    # (though usually receive is later)
    df.sort_values(["time", "di"], inplace=True)

    # Dictionary to track packets in flight per flow
    # Key: flow_id
    # Value: List of tuples (u, v, start_time)
    flow_queues: Dict[int, List[Tuple[int, int, int]]] = {}
    
    segments: List[Segment] = []

    for _, row in df.iterrows():
        flow = int(row["stream"])
        di = int(row["di"])
        t = int(row["time"])

        if di == 1:
            # SEND Event: Packet enters a link.
            # We record the Link (u, v) and the Start Time.
            u, v = row["u_v"]
            flow_queues.setdefault(flow, []).append((u, v, t))
            
        elif di == 0:
            # ARRIVE Event: Packet finishes a link.
            # We assume FIFO ordering for the flow. 
            # We POP the matching start event to get the correct Source/Dest.
            q = flow_queues.get(flow)
            if q:
                # Retrieve the 'u, v' from when the packet STARTED the hop.
                # We ignore the 'link' column in this current row because 
                # the log might have already updated it to the next hop.
                src, dst, t0 = q.pop(0)
                
                # Ensure strictly positive duration so it renders
                if t <= t0: t = t0 + 1
                
                segments.append(Segment(flow=flow, u=src, v=dst, t0_ns=t0, t1_ns=t))
            else:
                # We found an arrival (di=0) without a matching send (di=1).
                # This can happen if the simulation log starts mid-flow.
                pass

    print(f"Parsed {len(segments)} segments.")
    return segments

def _end_stations(G: nx.DiGraph) -> Set[int]:
    UG = G.to_undirected()
    return {n for n, deg in UG.degree() if deg == 1}


def align_to_axes(pos: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
    """
    Rotates the layout using PCA to align the principal axes with X and Y.
    This fixes 'tilted' grid layouts coming from spring_layout.
    """
    if not pos:
        return pos
    
    # Convert dict to matrix
    keys = sorted(pos.keys())
    coords = np.array([pos[k] for k in keys])
    
    # Center data
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    
    # PCA: Compute Covariance Matrix and Eigendecomposition
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eig(cov)
    
    # Sort eigenvectors by eigenvalues (largest variance first)
    idx = evals.argsort()[::-1]
    evecs = evecs[:, idx]
    
    # Rotate coordinates
    rotated = np.dot(centered, evecs)
    
    # Rebuild dict
    new_pos = {k: (float(rotated[i, 0]), float(rotated[i, 1])) for i, k in enumerate(keys)}
    return new_pos


def compute_aligned_grid_layout(G: nx.DiGraph) -> Dict[int, Tuple[float, float]]:
    # 1. Use Kamada-Kawai (better for mesh structures than spring)
    try:
        # Requires scipy, fallback to spring if missing
        pos = nx.kamada_kawai_layout(G.to_undirected(), scale=1.0)
    except:
        pos = nx.spring_layout(G, seed=42, iterations=200)

    # 2. Rotate to align with X/Y axes
    pos = align_to_axes(pos)
    
    # 3. Scale and Snap
    scale_factor = max(len(G.nodes) * 1.5, 8)
    
    grid_pos = {}
    occupied = set()

    # Prioritize switches (usually central) for better packing
    ends = _end_stations(G)
    nodes_sorted = sorted(list(G.nodes), key=lambda x: 1 if x in ends else 0)

    for n in nodes_sorted:
        x, y = pos[n]
        gx, gy = int(round(x * scale_factor)), int(round(y * scale_factor))
        
        # Spiral search to handle collisions
        # Searches (0,0), (1,0), (0,1), (-1,0), etc.
        r = 0
        found = False
        while not found:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r: continue # only check outer shell
                    candidate = (gx + dx, gy + dy)
                    if candidate not in occupied:
                        grid_pos[n] = (float(candidate[0]), float(candidate[1]))
                        occupied.add(candidate)
                        found = True
                        break
                if found: break
            r += 1
            
    return grid_pos


def animate(G: nx.DiGraph,
            segments: List[Segment],
            flow_to_src: Dict[int, int],
            flow_to_dsts: Dict[int, List[int]],
            ns_per_sec: float = 1000.0,
            fps: int = 30,
            save_path: Optional[str] = None) -> None:
    
    ends = _end_stations(G)
    switches = set(G.nodes()) - ends

    # --- Layout ---
    pos = compute_aligned_grid_layout(G)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("TSN Animation (Straight & Aligned)")
    ax.set_aspect('equal')
    ax.axis('off')

    # --- Draw Static Edges (Straight Lines) ---
    UG = G.to_undirected()
    
    for u, v in UG.edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#bbbbbb", linewidth=2.0, zorder=1)

    # --- Draw Nodes ---
    ends_list = list(ends)
    switches_list = list(switches)
    
    if ends_list:
        nx.draw_networkx_nodes(
            G, pos, nodelist=ends_list, ax=ax, node_size=350,
            node_color='#2ca02c', edgecolors='#2e6b2e', linewidths=1.0, label='End-station'
        )
    if switches_list:
        nx.draw_networkx_nodes(
            G, pos, nodelist=switches_list, ax=ax, node_size=280,
            node_color='#1f77b4', edgecolors='#274d70', linewidths=1.0, label='Switch', node_shape='s'
        )
    
    # Labels with slight offset
    label_pos = {n: (x, y + 0.3) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, ax=ax, font_size=8, font_color='#000000', font_weight='bold')

    # --- Animation ---
    flow_ids = sorted({s.flow for s in segments})
    cmap = plt.get_cmap('tab20')
    flow_color = {fid: cmap(i % cmap.N) for i, fid in enumerate(flow_ids)}

    t0_anim = min(s.t0_anim(ns_per_sec) for s in segments) if segments else 0.0
    t1_anim = max(s.t1_anim(ns_per_sec) for s in segments) if segments else 0.0
    duration = t1_anim - t0_anim
    if duration <= 0: duration = 1.0
    total_frames = max(1, int(duration * fps) + 1)

    scat = ax.scatter([], [], s=45, c=[], alpha=1.0, zorder=3, edgecolors='white', linewidths=0.5)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))
        return (scat,)

    def update(frame: int):
        t_anim = t0_anim + frame / fps
        xs_p: List[float] = []
        ys_p: List[float] = []
        colors: List[Tuple] = []

        for seg in segments:
            start = seg.t0_anim(ns_per_sec)
            end = seg.t1_anim(ns_per_sec)
            
            if start <= t_anim <= end:
                if end == start:
                    progress = 1.0
                else:
                    progress = (t_anim - start) / (end - start)

                # Linear Interpolation (Straight Line)
                x0, y0 = pos[seg.u]
                x1, y1 = pos[seg.v]
                
                cx = x0 + (x1 - x0) * progress
                cy = y0 + (y1 - y0) * progress
                
                xs_p.append(cx)
                ys_p.append(cy)
                colors.append(flow_color.get(seg.flow, (0.5, 0.5, 0.5, 1.0)))

        if xs_p:
            offsets = np.column_stack([xs_p, ys_p])
            scat.set_offsets(offsets)
            scat.set_color(colors)
        else:
            scat.set_offsets(np.empty((0, 2)))
            scat.set_color([])
        return (scat,)

    ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, interval=1000 / fps)

    # Legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color='#2ca02c', label='End-station'),
        mpatches.Patch(color='#1f77b4', label='Switch'),
    ]
    if flow_ids:
        for f in flow_ids[:6]:
            handles.append(mpatches.Patch(color=flow_color[f], label=f"Flow {f}"))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()

    if save_path:
        print(f"Saving to {save_path}...")
        try:
            ani.save(save_path, fps=fps)
            print("Saved.")
        except Exception as e:
            print(f"Error saving: {e}")
            plt.show()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topo", default="demo/demo1/data/1_topo.csv")
    parser.add_argument("--tasks", default="demo/demo1/data/1_task.csv")
    parser.add_argument("--log", default="demo/demo1/simulation/log.csv")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ns-per-sec", type=float, default=1000.0)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    G = load_topology(args.topo)
    flow_to_src, flow_to_dsts = load_tasks(args.tasks)
    segments = build_segments(args.log)

    if not segments:
        print("No segments found.")
        return

    animate(G, segments, flow_to_src, flow_to_dsts, ns_per_sec=args.ns_per_sec, fps=args.fps, save_path=args.save)


if __name__ == "__main__":
    main()