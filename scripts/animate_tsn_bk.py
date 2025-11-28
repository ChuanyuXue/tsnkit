import argparse
import ast
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import pandas as pd

# Try to import IPython for Notebook rendering
try:
    from IPython.display import HTML
    from IPython import get_ipython
except ImportError:
    HTML = None
    get_ipython = None

class TSNProVisualizer:
    def __init__(self, topo_path, task_path, log_path):
        self.G = self._load_topology(topo_path)
        self.flow_src, self.flow_dsts = self._load_tasks(task_path)
        self.segments = self._build_segments(log_path)
        
        # --- Professional Layout & Style ---
        # Kamada-Kawai looks organic and professional for network topos
        self.pos = nx.kamada_kawai_layout(self.G.to_undirected())
        
        self.colors = {
            'bg': '#1e1e1e',       # Dark Grey (IDE style)
            'edge': '#444444',     # Subtle edge
            'sw': '#00BFFF',       # Deep Sky Blue (Switch)
            'es': '#32CD32',       # Lime Green (End Station)
            'text': '#E0E0E0'      # Off-white
        }

    def _parse_link(self, s):
        s = s.strip().strip('"').strip()
        if not (s.startswith("(") and s.endswith(")")):
            # Handle potential variation in log format
            parts = s.split(',')
            if len(parts) == 2: return int(parts[0]), int(parts[1])
            raise ValueError(f"Bad link format: {s}")
        u_str, v_str = s[1:-1].split(",")
        return int(u_str.strip()), int(v_str.strip())

    def _load_topology(self, path):
        df = pd.read_csv(path)
        G = nx.DiGraph()
        for _, row in df.iterrows():
            u, v = self._parse_link(row["link"])
            G.add_edge(u, v)
        return G

    def _load_tasks(self, path):
        df = pd.read_csv(path)
        flow_to_src = {}
        flow_to_dsts = {}
        for _, row in df.iterrows():
            fid = int(row["stream"])
            flow_to_src[fid] = int(row["src"])
            dst_raw = row["dst"]
            try:
                dsts = list(map(int, ast.literal_eval(dst_raw)))
            except:
                dsts = [int(dst_raw)]
            flow_to_dsts[fid] = dsts
        return flow_to_src, flow_to_dsts

    def _build_segments(self, log_path):
        df = pd.read_csv(log_path)
        df["u_v"] = df["link"].map(self._parse_link)
        df.sort_values(["time", "di"], inplace=True)
        
        flow_queues = {}
        segments = []
        
        for _, row in df.iterrows():
            flow = int(row["stream"])
            di = int(row["di"])
            t = int(row["time"])
            
            if di == 1: # Enqueue (Send)
                u, v = row["u_v"]
                flow_queues.setdefault(flow, []).append((u, v, t))
            elif di == 0: # Dequeue (Arrive)
                q = flow_queues.get(flow)
                if q:
                    src, dst, t0 = q.pop(0)
                    if t <= t0: t = t0 + 100 # Visual padding
                    segments.append({'flow': flow, 'u': src, 'v': dst, 't0': t0, 't1': t})
        return segments

    def _is_notebook(self):
        """Detect if running inside Jupyter/Colab."""
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            return False
        except NameError:
            return False

    def show(self, ns_per_sec=1000.0, fps=30, speed=1.0):
        """
        Main entry point. Detects environment and renders accordingly.
        """
        if not self.segments:
            print("Error: No packet segments found in log.")
            return

        # 1. Setup Figure
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(self.colors['bg'])
        ax.set_facecolor(self.colors['bg'])
        ax.axis('off')
        
        # 2. Draw Static Elements
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, edge_color=self.colors['edge'], 
                               width=1.5, arrowsize=10)
        
        ends = [n for n in self.G.nodes if self.G.degree(n) == 1]
        switches = [n for n in self.G.nodes if n not in ends]
        
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=switches, ax=ax, 
                               node_size=300, node_color=self.colors['sw'], node_shape='s', label='Switch')
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=ends, ax=ax, 
                               node_size=350, node_color=self.colors['es'], node_shape='o', label='End Station')
        
        nx.draw_networkx_labels(self.G, self.pos, {n:str(n) for n in self.G.nodes}, 
                                font_color=self.colors['text'], font_size=8, ax=ax)

        # 3. HUD (Heads Up Display)
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color=self.colors['text'], 
                            fontsize=12, fontfamily='monospace', weight='bold')
        ax.legend(loc='upper right', facecolor=self.colors['bg'], edgecolor=self.colors['edge'], labelcolor=self.colors['text'])

        # 4. Animation Logic
        min_t = min(s['t0'] for s in self.segments)
        max_t = max(s['t1'] for s in self.segments)
        total_dur = (max_t - min_t)
        
        # Calculate Frames
        # Duration in seconds * fps
        real_duration_sec = (total_dur / ns_per_sec) / speed
        total_frames = int(real_duration_sec * fps)
        if total_frames < 1: total_frames = 1

        cmap = plt.get_cmap('tab10')
        scat = ax.scatter([], [], s=100, zorder=5, edgecolors='white', linewidths=0.8)

        def init():
            scat.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            return scat, time_text

        def update(frame):
            progress = frame / total_frames
            curr_time = min_t + (progress * total_dur)
            
            # Filter active packets
            active = [s for s in self.segments if s['t0'] <= curr_time <= s['t1']]
            
            x, y, c = [], [], []
            for s in active:
                dur = s['t1'] - s['t0']
                p = (curr_time - s['t0']) / dur if dur > 0 else 1.0
                
                u_pos = np.array(self.pos[s['u']])
                v_pos = np.array(self.pos[s['v']])
                curr_pos = u_pos + (v_pos - u_pos) * p
                
                x.append(curr_pos[0])
                y.append(curr_pos[1])
                c.append(cmap(s['flow'] % 10))
            
            time_text.set_text(f"TIME: {int(curr_time):08d} ns")
            
            if x:
                scat.set_offsets(np.column_stack([x, y]))
                scat.set_color(c)
            else:
                scat.set_offsets(np.empty((0, 2)))
            return scat, time_text

        ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, interval=1000/fps)

        # 5. Output Selection
        if self._is_notebook():
            print("Rendering for Jupyter Notebook...")
            plt.close(fig) # Prevent double plotting
            return HTML(ani.to_jshtml())
        else:
            print("Rendering for Desktop Window...")
            plt.show()

if __name__ == "__main__":
    # Command Line Interface
    parser = argparse.ArgumentParser(description="TSN Demo Visualizer")
    parser.add_argument("--topo", default="demo/demo1/data/1_topo.csv")
    parser.add_argument("--tasks", default="demo/demo1/data/1_task.csv")
    parser.add_argument("--log", default="demo/demo1/simulation/log.csv")
    parser.add_argument("--speed", type=float, default=1.0)
    
    args = parser.parse_args()
    
    viz = TSNProVisualizer(args.topo, args.tasks, args.log)
    viz.show(speed=args.speed)