import argparse
import ast
import bisect  # Efficient sorting/searching for dynamic counters
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict

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
        
        # Store LISTS of timestamps instead of just counts
        self.tx_times = defaultdict(list)
        self.rx_times = defaultdict(list)
        
        self._load_event_times(log_path)
        self.segments = self._build_segments(log_path)
        
        # --- Professional Layout & Style ---
        self.pos = nx.kamada_kawai_layout(self.G.to_undirected())
        
        self.colors = {
            'bg': '#1e1e1e',       # Dark Grey
            'edge': '#444444',     # Subtle edge
            'sw': '#00BFFF',       # Deep Sky Blue
            'es': '#32CD32',       # Lime Green
            'text': '#FFFFFF'      # White text for readability
        }

    def _parse_link(self, s):
        s = s.strip().strip('"').strip()
        if not (s.startswith("(") and s.endswith(")")):
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

    def _load_event_times(self, log_path):
        """
        Parses the log to store sorted lists of event timestamps for each node.
        This allows us to calculate dynamic counts at any specific time t.
        """
        try:
            df = pd.read_csv(log_path)
        except FileNotFoundError:
            print(f"Warning: Log file '{log_path}' not found.")
            return

        for _, row in df.iterrows():
            try:
                u, v = self._parse_link(row["link"])
                di = int(row["di"])
                t = int(row["time"])
                
                if di == 1: # TX (Send) happens at u
                    self.tx_times[u].append(t)
                elif di == 0: # RX (Receive) happens at v
                    self.rx_times[v].append(t)
            except (ValueError, KeyError):
                continue
        
        # Sort lists for efficient bisect searching
        for n in self.tx_times: self.tx_times[n].sort()
        for n in self.rx_times: self.rx_times[n].sort()

    def _build_segments(self, log_path):
        try:
            df = pd.read_csv(log_path)
        except FileNotFoundError:
            return []

        df["u_v"] = df["link"].map(self._parse_link)
        df.sort_values(["time", "di"], inplace=True)
        
        flow_queues = {}
        segments = []
        
        for _, row in df.iterrows():
            flow = int(row["stream"])
            di = int(row["di"])
            t = int(row["time"])
            
            if di == 1: # Enqueue
                u, v = row["u_v"]
                flow_queues.setdefault(flow, []).append((u, v, t))
            elif di == 0: # Dequeue
                q = flow_queues.get(flow)
                if q:
                    src, dst, t0 = q.pop(0)
                    if t <= t0: t = t0 + 100 
                    segments.append({'flow': flow, 'u': src, 'v': dst, 't0': t0, 't1': t})
        return segments

    def _is_notebook(self):
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell': return True 
            elif shell == 'TerminalInteractiveShell': return False
            return False
        except NameError:
            return False

    def show(self, ns_per_sec=1000.0, fps=30, speed=1.0):
        if not self.segments:
            print("Error: No packet segments found.")
            return

        # 1. Setup Figure
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor(self.colors['bg'])
        ax.set_facecolor(self.colors['bg'])
        ax.axis('off')
        
        # 2. Draw Static Elements
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, edge_color=self.colors['edge'], 
                               width=1.5, arrowsize=10)
        
        ends = [n for n in self.G.nodes if self.G.degree(n) == 2]
        switches = [n for n in self.G.nodes if n not in ends]
        
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=switches, ax=ax, 
                               node_size=600, node_color=self.colors['sw'], node_shape='s', label='Switch')
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=ends, ax=ax, 
                               node_size=600, node_color=self.colors['es'], node_shape='s', label='End Station')

    def show(self, ns_per_sec=10000.0, fps=30, speed=1.0):
        if not self.segments:
            print("Error: No packet segments found.")
            return

        # 1. Setup Figure
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor(self.colors['bg'])
        ax.set_facecolor(self.colors['bg'])
        ax.axis('off')
        
        # 2. Draw Static Elements (Edges & Nodes)
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, edge_color=self.colors['edge'], 
                               width=1.5, arrowsize=10)
        
        ends = [n for n in self.G.nodes if self.G.degree(n) == 2]
        switches = [n for n in self.G.nodes if n not in ends]
        
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=switches, ax=ax, 
                               node_size=600, node_color=self.colors['sw'], node_shape='s', label='Switch')
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=ends, ax=ax, 
                               node_size=600, node_color=self.colors['es'], node_shape='s', label='End Station')

        # --- LABEL SPLIT STRATEGY ---

        # A. Draw STATIC Node IDs (INSIDE the box)
        # We use self.pos directly so they sit in the center
        nx.draw_networkx_labels(self.G, self.pos, labels={n: str(n) for n in self.G.nodes},
                                font_color='black', font_weight='bold', font_size=9, ax=ax)

        # B. Initialize DYNAMIC Counters (OUTSIDE the box)
        # 1. Calculate shifted positions (Upwards by 0.1)
        offset_y = 0.1
        label_pos = {n: (x, y + offset_y) for n, (x, y) in self.pos.items()}

        # 2. Create placeholder text for counters
        initial_counters = {n: "TX:0\nRX:0" for n in self.G.nodes}
        
        # 3. Draw the counters. We capture the returned dict to update them later.
        # We use a smaller font and a background box for readability.
        counter_text_objs = nx.draw_networkx_labels(
            self.G, 
            pos=label_pos,
            labels=initial_counters, 
            font_color=self.colors['text'], 
            font_size=7, 
            font_family='monospace',
            ax=ax,
            bbox=dict(boxstyle="round,pad=0.2", fc=self.colors['bg'], ec=self.colors['edge'], alpha=0.6)
        )

        # 3. HUD
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color=self.colors['text'], 
                            fontsize=14, fontfamily='monospace', weight='bold')
        ax.legend(loc='upper right', facecolor=self.colors['bg'], edgecolor=self.colors['edge'], labelcolor=self.colors['text'])

        # 4. Animation Logic
        min_t = min(s['t0'] for s in self.segments)
        max_t = max(s['t1'] for s in self.segments)
        total_dur = (max_t - min_t)
        
        real_duration_sec = (total_dur / ns_per_sec) / speed
        frames_per_cycle = int(real_duration_sec * fps)
        if frames_per_cycle < 1: frames_per_cycle = 1

        cmap = plt.get_cmap('tab10')
        scat = ax.scatter([], [], s=100, zorder=5, edgecolors='white', linewidths=0.8)

        def init():
            scat.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            # Return scat, time_text, and all counter text objects
            return [scat, time_text] + list(counter_text_objs.values())

        def update(frame):
            # --- Non-Stop Time Logic ---
            cycle_idx = frame // frames_per_cycle
            frame_in_cycle = frame % frames_per_cycle
            
            progress = frame_in_cycle / frames_per_cycle
            local_curr_time = min_t + (progress * total_dur)
            global_display_time = local_curr_time + (cycle_idx * total_dur)

            # --- Update Packets (Scatter) ---
            active = [s for s in self.segments if s['t0'] <= local_curr_time <= s['t1']]
            x, y, c = [], [], []
            for s in active:
                dur = s['t1'] - s['t0']
                p = (local_curr_time - s['t0']) / dur if dur > 0 else 1.0
                u_pos = np.array(self.pos[s['u']])
                v_pos = np.array(self.pos[s['v']])
                curr_pos = u_pos + (v_pos - u_pos) * p
                x.append(curr_pos[0])
                y.append(curr_pos[1])
                c.append(cmap(s['flow'] % 10))
            
            if x:
                scat.set_offsets(np.column_stack([x, y]))
                scat.set_color(c)
            else:
                scat.set_offsets(np.empty((0, 2)))

            # --- Update DYNAMIC Counters Only ---
            for n, text_obj in counter_text_objs.items():
                base_tx = len(self.tx_times[n]) * cycle_idx
                base_rx = len(self.rx_times[n]) * cycle_idx
                
                curr_tx = bisect.bisect_right(self.tx_times[n], local_curr_time)
                curr_rx = bisect.bisect_right(self.rx_times[n], local_curr_time)
                
                total_tx = base_tx + curr_tx
                total_rx = base_rx + curr_rx
                
                text_obj.set_text(f"TX:{total_tx}\nRX:{total_rx}")

            time_text.set_text(f"TIME: {int(global_display_time):08d} ns")
            return [scat, time_text] + list(counter_text_objs.values())

        # 5. Output Selection
        if self._is_notebook():
            print("Rendering for Jupyter (Limited to 10 cycles)...")
            ani = animation.FuncAnimation(fig, update, frames=frames_per_cycle * 10, 
                                          init_func=init, blit=True, interval=1000/fps)
            plt.close(fig)
            return HTML(ani.to_jshtml())
        else:
            print("Rendering for Desktop (Infinite Loop)...")
            ani = animation.FuncAnimation(fig, update, frames=None, 
                                          init_func=init, blit=True, interval=1000/fps,
                                          cache_frame_data=False)
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSN Demo Visualizer")
    parser.add_argument("--topo", default="data/1_topo.csv")
    parser.add_argument("--tasks", default="data/1_task.csv")
    parser.add_argument("--log", default="simulation/log.csv")
    parser.add_argument("--speed", type=float, default=1.0)
    
    args = parser.parse_args()
    
    viz = TSNProVisualizer(args.topo, args.tasks, args.log)
    viz.show(speed=args.speed)