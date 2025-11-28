import argparse
import ast
import bisect
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict, deque

# Try to import IPython for Notebook rendering
try:
    from IPython.display import HTML
    from IPython import get_ipython
except ImportError:
    HTML = None
    get_ipython = None

class TSNProVisualizer:
    def __init__(self, topo_path, task_path, log_path):
        self.colors = {
            'bg': '#1e1e1e',       
            'panel_bg': '#252526', 
            'edge': '#444444',
            'sw': '#00BFFF',
            'es': '#32CD32',
            'text': '#E0E0E0',
            'grid': '#333333'
        }
        
        self.G = self._load_topology(topo_path)
        self.flow_src, self.flow_dsts = self._load_tasks(task_path)
        
        # Data containers
        self.tx_times = defaultdict(list)
        self.rx_times = defaultdict(list)
        
        # Load raw events for counters
        self._load_event_times(log_path)
        # Load segments for animation
        self.segments = self._build_segments(log_path)
        
        # Precompute Logs and Delay Stats
        self.log_entries = []     # legacy preformatted strings (per-cycle)
        self.events = []          # structured events for continuous-time logging
        self.event_times = []     # event times for bisect (ns)
        self.delay_data = {}      
        self._precompute_stats(log_path)
        
        # Layout & Position
        self.pos = nx.kamada_kawai_layout(self.G.to_undirected())

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
                # Handle cases like "[3]" or "3"
                if isinstance(dst_raw, str) and "[" in dst_raw:
                    dsts = list(map(int, ast.literal_eval(dst_raw)))
                else:
                    dsts = [int(dst_raw)]
            except:
                dsts = [int(dst_raw)] if pd.notna(dst_raw) else []
            flow_to_dsts[fid] = dsts
        return flow_to_src, flow_to_dsts

    def _load_event_times(self, log_path):
        try:
            df = pd.read_csv(log_path)
        except FileNotFoundError:
            return
        for _, row in df.iterrows():
            try:
                u, v = self._parse_link(row["link"])
                di = int(row["di"])
                t = int(row["time"])
                if di == 1: self.tx_times[u].append(t)
                elif di == 0: self.rx_times[v].append(t)
            except: continue
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

    def _precompute_stats(self, log_path):
        """
        Robustly parses log for Delay and Text entries.
        """
        try:
            df = pd.read_csv(log_path)
            df["u_v"] = df["link"].map(self._parse_link)
            df.sort_values("time", inplace=True)
        except:
            print("Error reading log for stats.")
            return

        # 1. Build Delay Stats (End-to-End)
        # We track packets leaving Source and hitting Destination
        tx_queues = defaultdict(list) # flow -> [timestamps]
        
        for _, row in df.iterrows():
            f = int(row["stream"])
            t = int(row["time"])
            u, v = row["u_v"]
            di = int(row["di"])
            
            # Create structured event for continuous-time logging
            evt = "Send" if di == 1 else "Recv"
            self.events.append({'time': t, 'flow': f, 'evt': evt, 'u': int(u), 'v': int(v)})
            self.event_times.append(t)
            # Also keep a preformatted message (per-cycle) for fallback
            ts_str = f"[{t/1e9:.9f}]"
            self.log_entries.append((t, f"{ts_str} Flw{f}: {evt} {u}->{v}"))

            # Delay Logic
            src_node = self.flow_src.get(f)
            dst_nodes = self.flow_dsts.get(f, [])
            
            # --- FIX: Strict Integer Matching ---
            # Ensure we are comparing ints to ints
            u_int = int(u)
            v_int = int(v)
            
            # Record Source Transmission
            if di == 1 and u_int == src_node:
                tx_queues[f].append(t)
            
            # Record Destination Reception
            if di == 0 and v_int in dst_nodes:
                if tx_queues[f]:
                    start_t = tx_queues[f].pop(0)
                    delay = (t - start_t) / 1000.0 # Convert ns to us (microseconds)
                    
                    if f not in self.delay_data:
                        self.delay_data[f] = {'times': [], 'delays': []}
                    
                    self.delay_data[f]['times'].append(t)
                    self.delay_data[f]['delays'].append(delay)
        
        if not self.delay_data:
            print("WARNING: No end-to-end delays detected. Check if 'src'/'dst' in task.csv match 'link' IDs in log.csv.")

    def _is_notebook(self):
        try:
            shell = get_ipython().__class__.__name__
            return shell == 'ZMQInteractiveShell'
        except NameError:
            return False

    def show(self, ns_per_sec=10000.0, fps=30, speed=1.0):
        if not self.segments:
            print("Error: No packet segments found.")
            return

        # --- 1. Setup Figure ---
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 9))
        fig.patch.set_facecolor(self.colors['bg'])
        
        gs = GridSpec(2, 3, figure=fig, width_ratios=[3, 3, 2], height_ratios=[1, 1])
        
        # A. Topology Panel
        ax_topo = fig.add_subplot(gs[:, :2])
        ax_topo.set_facecolor(self.colors['bg'])
        ax_topo.axis('off')

        # B. Log Panel
        ax_log = fig.add_subplot(gs[0, 2])
        ax_log.set_facecolor(self.colors['panel_bg'])
        ax_log.set_title("Events Log", color='white', fontsize=10, loc='left', pad=10)
        ax_log.axis('off')
        
        log_text_obj = ax_log.text(0.02, 0.95, "Initializing...", transform=ax_log.transAxes, 
                                   va='top', ha='left', fontsize=9, 
                                   fontfamily='monospace', color='#00FF00', linespacing=1.5)

        # C. Delay Panel
        ax_delay = fig.add_subplot(gs[1, 2])
        ax_delay.set_facecolor(self.colors['panel_bg'])
        ax_delay.set_title("End-to-End Delay (us)", color='white', fontsize=10, loc='left', pad=10)
        ax_delay.grid(True, color=self.colors['grid'], linestyle='--', linewidth=0.5)
        ax_delay.tick_params(colors=self.colors['text'], labelsize=8)
        for spine in ax_delay.spines.values():
            spine.set_edgecolor(self.colors['edge'])
            
        # Initialize Delay Lines
        flow_lines = {}
        cmap = plt.get_cmap('tab10')
        all_delays = []
        all_delay_times = []
        for f, data in self.delay_data.items():
            line, = ax_delay.plot([], [], label=f"Flow {f}", color=cmap(f%10), linewidth=1.0, marker='o', markersize=3, zorder=3)
            flow_lines[f] = line
            all_delays.extend(data['delays'])
            all_delay_times.extend(data['times'])

        # Auto-scale Y axis once based on max delay found
        if all_delays:
            max_delay = max(all_delays)
            ax_delay.set_ylim(0, max_delay * 1.2)
        else:
            ax_delay.set_ylim(0, 100) # Default fallback
        
        # Initialize X axis window up-front so data are visible even with blitting
        # Use a 20% window of the total simulation duration
        # (update() will keep sliding this window as time advances)
        # Compute from segments time range below

        # --- 2. Static Topology Draw ---
        nx.draw_networkx_edges(self.G, self.pos, ax=ax_topo, edge_color=self.colors['edge'], width=1.5, arrowsize=10)
        ends = [n for n in self.G.nodes if self.G.degree(n) == 2]
        switches = [n for n in self.G.nodes if n not in ends]
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=switches, ax=ax_topo, 
                               node_size=600, node_color=self.colors['sw'], node_shape='s')
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=ends, ax=ax_topo, 
                               node_size=600, node_color=self.colors['es'], node_shape='s')
        nx.draw_networkx_labels(self.G, self.pos, labels={n: str(n) for n in self.G.nodes},
                                font_color='black', font_weight='bold', font_size=9, ax=ax_topo)

        # Counters
        offset_y = 0.1
        label_pos = {n: (x, y + offset_y) for n, (x, y) in self.pos.items()}
        counter_text_objs = nx.draw_networkx_labels(
            self.G, pos=label_pos, labels={n: "TX:0\nRX:0" for n in self.G.nodes}, 
            font_color=self.colors['text'], font_size=7, font_family='monospace', ax=ax_topo,
            bbox=dict(boxstyle="round,pad=0.2", fc=self.colors['bg'], ec=self.colors['edge'], alpha=0.6)
        )

        time_text = ax_topo.text(0.02, 0.95, '', transform=ax_topo.transAxes, color=self.colors['text'], 
                                 fontsize=14, fontfamily='monospace', weight='bold')

        # --- 3. Animation Logic (Infinite-Cycle Like visual2) ---
        
        # Calculate global time range from logs
        min_t = min(s['t0'] for s in self.segments)
        max_t = max(s['t1'] for s in self.segments)
        total_dur_ns = max_t - min_t
        # Set initial X-window for delay chart before animation starts
        # Anchor to earliest delay time if available, otherwise to segment start
        initial_window = total_dur_ns * 0.2 if total_dur_ns > 0 else 1
        delay_start = min(all_delay_times) if all_delay_times else min_t
        ax_delay.set_xlim(delay_start, delay_start + initial_window)
        
        # Calculate simulation time advance per frame in nanoseconds
        # This avoids rounding issues from integer frames_per_cycle and keeps motion smooth.
        real_duration_sec = (total_dur_ns / ns_per_sec) / speed
        step_ns = (ns_per_sec / fps) * speed  # simulation nanoseconds advanced per rendered frame
        if step_ns <= 0:
            step_ns = max(1, total_dur_ns // max(1, int(real_duration_sec * fps)) if real_duration_sec > 0 else 1)
        
        scat = ax_topo.scatter([], [], s=100, zorder=5, edgecolors='white', linewidths=0.8)

        def init():
            scat.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            log_text_obj.set_text('')
            for line in flow_lines.values():
                line.set_data([], [])
            return [scat, time_text, log_text_obj] + list(counter_text_objs.values()) + list(flow_lines.values())

        def update(frame):
            # Use a monotonic simulation time derived from frame count
            sim_elapsed = frame * step_ns
            cycle_idx = int(sim_elapsed // total_dur_ns) if total_dur_ns > 0 else 0
            local_offset = sim_elapsed % total_dur_ns if total_dur_ns > 0 else 0
            # Local time within the current cycle and global display time
            local_curr_time = min_t + local_offset
            global_display_time = min_t + sim_elapsed
            
            # 1. Update Packets
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

            # 2. Update Counters (accumulate across cycles like visual2)
            for n, text_obj in counter_text_objs.items():
                base_tx = len(self.tx_times[n]) * cycle_idx
                base_rx = len(self.rx_times[n]) * cycle_idx
                curr_tx = bisect.bisect_right(self.tx_times[n], local_curr_time)
                curr_rx = bisect.bisect_right(self.rx_times[n], local_curr_time)
                total_tx = base_tx + curr_tx
                total_rx = base_rx + curr_rx
                text_obj.set_text(f"TX:{total_tx}\nRX:{total_rx}")

            # 3. Update Log Panel (Scroll last 15 lines) with continuous timestamps
            recent_lines = []
            if self.events:
                # Index within this cycle
                pos = bisect.bisect_right(self.event_times, local_curr_time) - 1
                c = cycle_idx
                i = pos
                while c >= 0 and len(recent_lines) < 15:
                    while i >= 0 and len(recent_lines) < 15:
                        ev = self.events[i]
                        t_disp = ev['time'] + c * total_dur_ns
                        recent_lines.append(f"[{t_disp/1e9:.9f}] Flw{ev['flow']}: {ev['evt']} {ev['u']}->{ev['v']}")
                        i -= 1
                    c -= 1
                    i = len(self.events) - 1
                recent_lines.reverse()
                log_text_obj.set_text("\n".join(recent_lines))
            else:
                # Fallback to precomputed per-cycle strings
                log_idx = bisect.bisect_right(self.log_entries, (local_curr_time, "zzzz"))
                recent_logs = self.log_entries[max(0, log_idx-15):log_idx]
                log_text_obj.set_text("\n".join([entry[1] for entry in recent_logs]))
            
            # 4. Update Delay Chart (Sliding Window) with continuous timestamps
            # Build only the points within the current global time window to keep drawing lightweight
            window_size = total_dur_ns * 0.2  # 20% window
            global_now = global_display_time
            win_start = max(delay_start, global_now - window_size)
            for f, line in flow_lines.items():
                f_times = self.delay_data[f]['times']
                f_delays = self.delay_data[f]['delays']
                if not f_times:
                    line.set_data([], [])
                    continue
                xt = []
                yt = []
                # Include a bounded number of prior cycles likely to intersect the window
                # Compute how many full cycles can fit in the window (usually 0 or 1)
                cycles_back = int(np.ceil(window_size / total_dur_ns)) if total_dur_ns > 0 else 0
                k_start = max(0, cycle_idx - cycles_back - 1)
                for k in range(k_start, cycle_idx):
                    off = k * total_dur_ns
                    # Full past cycles
                    for t, d in zip(f_times, f_delays):
                        tt = t + off
                        if tt >= win_start:
                            xt.append(tt)
                            yt.append(d)
                # Current cycle up to local time
                idx = bisect.bisect_right(f_times, local_curr_time)
                if idx > 0:
                    off = cycle_idx * total_dur_ns
                    for t, d in zip(f_times[:idx], f_delays[:idx]):
                        tt = t + off
                        if tt >= win_start:
                            xt.append(tt)
                            yt.append(d)
                if xt:
                    line.set_data(np.asarray(xt), np.asarray(yt))
                else:
                    line.set_data([], [])
            # Slide X-Axis window strictly by global time
            ax_delay.set_xlim(win_start, global_now)

            time_text.set_text(f"TIME: {int(global_display_time):08d} ns")
            time_text.set_color('#00FF00')
            return [scat, time_text, log_text_obj] + list(counter_text_objs.values()) + list(flow_lines.values())

        print(f"Starting Animation: step {step_ns:.2f} ns/frame, approx {real_duration_sec:.2f} sec/cycle.")
        
        if self._is_notebook():
            ani = animation.FuncAnimation(fig, update, frames=int(max(1, (total_dur_ns/step_ns)))*10, 
                                          init_func=init, blit=False, interval=1000/fps)
            plt.close(fig)
            return HTML(ani.to_jshtml())
        else:
            ani = animation.FuncAnimation(fig, update, frames=None, 
                                          init_func=init, blit=False, interval=1000/fps,
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
