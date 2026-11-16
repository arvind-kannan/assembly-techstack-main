# examples/navigate_disassembly_graph.py
"""
Interactive local navigator for a disassembly graph produced by
DisassemblyGraphAgent (see extract_disassembly_graph.py).

Rather than rendering the whole (potentially huge) graph at once, this shows
one node -- the "current" assembly state -- together with its immediate
neighborhood: the actions (edges) that got you here, and the actions
available from here. Click an outgoing action (button or node in the plot)
to move forward; click "Back" (or press backspace) to undo.

Usage:
    python examples/navigate_disassembly_graph.py

Controls:
    - Click an action button, or click a node in the graph view, to navigate.
    - Number keys 1-9 also trigger the corresponding numbered action.
    - "Back" button or Backspace/'b' key: undo one navigation step.
    - "Restart" button: jump back to the fully-assembled root state.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "examples"))

import matplotlib.pyplot as plt
import networkx as nx
from build_env import build_env
from matplotlib.widgets import Button

from disassembly_pomdp.graph_extraction import (
    DisassemblyGraphAgent,
    load_graph,
    save_graph,
)


class GraphNavigator:
    """Interactive one-node-at-a-time viewer over a disassembly MultiDiGraph."""

    def __init__(self, graph: nx.MultiDiGraph, root):
        self.graph = graph
        self.root = root
        self.current = root
        self.history: list = []
        self._last_pos: dict = {}
        self._action_widgets: list = []

        self.fig = plt.figure(figsize=(9.5, 8.5))
        try:
            self.fig.canvas.manager.set_window_title("Disassembly graph navigator")
        except Exception:
            pass

        self.ax_graph = self.fig.add_axes([0.05, 0.30, 0.9, 0.64])
        self.ax_info = self.fig.add_axes([0.05, 0.21, 0.9, 0.07])
        self.ax_info.axis("off")

        self.back_ax = self.fig.add_axes([0.05, 0.02, 0.14, 0.055])
        self.back_button = Button(self.back_ax, "◀ Back")
        self.back_button.on_clicked(self._go_back)

        self.restart_ax = self.fig.add_axes([0.21, 0.02, 0.16, 0.055])
        self.restart_button = Button(self.restart_ax, "↺ Restart")
        self.restart_button.on_clicked(self._restart)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click_graph)

        self._redraw()

    def show(self) -> None:
        plt.show()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _go_to(self, node_key):
        def handler(event=None):
            if node_key == self.current:
                return
            self.history.append(self.current)
            self.current = node_key
            self._redraw()

        return handler

    def _go_back(self, event=None):
        if self.history:
            self.current = self.history.pop()
            self._redraw()

    def _restart(self, event=None):
        if self.current != self.root:
            self.history = []
            self.current = self.root
            self._redraw()

    def _on_key(self, event):
        if event.key in ("backspace", "b"):
            self._go_back()
            return
        if event.key and event.key.isdigit():
            idx = int(event.key) - 1
            out_edges = list(self.graph.out_edges(self.current, keys=True, data=True))
            if 0 <= idx < len(out_edges):
                self._go_to(out_edges[idx][1])()

    def _on_click_graph(self, event):
        if event.inaxes is not self.ax_graph or event.xdata is None:
            return
        if not self._last_pos:
            return
        click = (event.xdata, event.ydata)
        best_node, best_d2 = None, None
        for node, (x, y) in self._last_pos.items():
            d2 = (x - click[0]) ** 2 + (y - click[1]) ** 2
            if best_d2 is None or d2 < best_d2:
                best_node, best_d2 = node, d2
        # Only accept clicks reasonably close to a node marker.
        if best_node is not None and best_d2 is not None and best_d2 < 0.02:
            self._go_to(best_node)()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _clear_action_widgets(self) -> None:
        for ax, _btn in self._action_widgets:
            self.fig.delaxes(ax)
        self._action_widgets = []

    def _redraw(self) -> None:
        node = self.current
        data = self.graph.nodes[node]

        out_edges = list(self.graph.out_edges(node, keys=True, data=True))
        in_edges = list(self.graph.in_edges(node, keys=True, data=True))

        # --- build a small ego graph: predecessors -> current -> successors ---
        H = nx.MultiDiGraph()
        H.add_node(node)
        for u, v, k, ed in in_edges:
            H.add_node(u)
            H.add_edge(u, v, key=k, **ed)
        for u, v, k, ed in out_edges:
            H.add_node(v)
            H.add_edge(u, v, key=k, **ed)

        preds = list(dict.fromkeys(u for u, _v, _k, _ed in in_edges))
        succs = list(dict.fromkeys(v for _u, v, _k, _ed in out_edges))

        pos = {node: (0.5, 0.5)}
        for i, p in enumerate(preds):
            pos[p] = ((i + 1) / (len(preds) + 1), 0.88)
        for i, s in enumerate(succs):
            pos[s] = ((i + 1) / (len(succs) + 1), 0.12)
        self._last_pos = pos

        self.ax_graph.clear()
        self.ax_graph.axis("off")
        self.ax_graph.set_xlim(-0.05, 1.05)
        self.ax_graph.set_ylim(0.0, 1.0)

        node_colors, node_labels = [], {}
        for n in H.nodes():
            nd = self.graph.nodes[n]
            if n == node:
                node_colors.append("#5b9bd5")
            elif nd.get("is_goal"):
                node_colors.append("#8fd18f")
            else:
                node_colors.append("#e8e8e8")
            node_labels[n] = f"{nd['n_parts_present']} parts"

        nx.draw_networkx_nodes(
            H,
            pos,
            ax=self.ax_graph,
            node_size=1500,
            node_color=node_colors,
            edgecolors="black",
        )
        nx.draw_networkx_labels(
            H, pos, labels=node_labels, ax=self.ax_graph, font_size=8
        )
        nx.draw_networkx_edges(
            H,
            pos,
            ax=self.ax_graph,
            arrows=True,
            arrowsize=15,
            connectionstyle="arc3,rad=0.12",
            node_size=1500,
        )

        out_edge_labels = {
            (u, v): f"[{i + 1}] {ed['label']}\n{ed['action_time']:.1f}s"
            for i, (u, v, _k, ed) in enumerate(out_edges)
        }
        nx.draw_networkx_edge_labels(
            H,
            pos,
            edge_labels=out_edge_labels,
            ax=self.ax_graph,
            font_size=7,
        )

        title = f"depth={data.get('depth')}   parts remaining={data['n_parts_present']}"
        if data.get("is_goal"):
            title += "   *** GOAL STATE ***"
        self.ax_graph.set_title(title, fontsize=11)

        # --- info line ---
        self.ax_info.clear()
        self.ax_info.axis("off")
        removed = data.get("parts_removed", [])
        self.ax_info.text(
            0.0,
            0.7,
            "Removed so far: " + (", ".join(removed) if removed else "(none)"),
            fontsize=9,
            va="top",
            wrap=True,
        )
        if not out_edges:
            self.ax_info.text(
                0.0,
                0.15,
                "No further actions from this state (dead end / fully disassembled).",
                fontsize=9,
                va="top",
                color="firebrick",
            )

        # --- action buttons (also serves as a legend for the [n] edge labels) ---
        self._clear_action_widgets()
        n = len(out_edges)
        if n > 0:
            btn_w = min(0.9 / n, 0.28)
            gap = 0.01
            total_w = n * btn_w + (n - 1) * gap
            x0 = max(0.02, 0.5 - total_w / 2)
            for i, (_u, v, _k, ed) in enumerate(out_edges):
                ax = self.fig.add_axes([x0 + i * (btn_w + gap), 0.10, btn_w, 0.055])
                btn = Button(ax, f"[{i + 1}] {ed['label']}\n{ed['action_time']:.1f}s")
                btn.label.set_fontsize(7)
                btn.on_clicked(self._go_to(v))
                self._action_widgets.append((ax, btn))

        self.back_button.ax.set_visible(bool(self.history))
        self.fig.canvas.draw_idle()


def _find_root(graph: nx.MultiDiGraph):
    for n, d in graph.nodes(data=True):
        if d.get("depth") == 0:
            return n
    # fallback: any node with no predecessors
    for n in graph.nodes():
        if graph.in_degree(n) == 0:
            return n
    return next(iter(graph.nodes()))


def main() -> None:
    cache_path = ROOT / "examples" / "runs" / "disassembly_graph" / "graph.pkl"

    if cache_path.exists():
        print(f"Loading cached graph from {cache_path}")
        graph = load_graph(cache_path)
    else:
        print("No cached graph found, building it now (this takes a few seconds)...")
        env = build_env()
        agent = DisassemblyGraphAgent(env)
        graph = agent.build_graph(verbose=True)
        save_graph(graph, cache_path)
        print(f"Cached graph to {cache_path}")

    root = _find_root(graph)
    nav = GraphNavigator(graph, root)
    nav.show()


if __name__ == "__main__":
    main()
