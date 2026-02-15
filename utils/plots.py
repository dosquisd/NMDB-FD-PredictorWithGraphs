from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .enums import Normalizer


def plot_graphs_result(
    df: pd.DataFrame,
    title: str = "",
    *,
    altitude_normalizer: Normalizer = Normalizer.NONE,
    fixed_node_size: Optional[float] = None,
    fig_transpose: bool = False,
    ncols_multiplier: float = 10.5,
    nrows_multiplier: float = 4.0,
    cmap: mpl.colors.Colormap = plt.cm.viridis,  # type: ignore
) -> Figure:
    unique_events = df["event_date"].unique()
    adjacency_methods = df["adjacency_method"].unique()

    nrows = len(unique_events)
    ncols = len(adjacency_methods)
    figsize = (ncols * ncols_multiplier, nrows * nrows_multiplier)
    if fig_transpose:
        nrows, ncols = ncols, nrows
        figsize = figsize[::-1]

    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
    )

    axes = axes.flatten()
    i = 0
    for event_date in unique_events:
        for adj_method in adjacency_methods:
            plot_data = df[
                (df["event_date"] == event_date)
                & (df["adjacency_method"] == adj_method)
            ]

            intensity = plot_data["intensity"].values[0]
            drop = plot_data["drop"].values[0]
            graph = plot_data["graph"].values[0]
            cutoff = graph.graph["metadata_cutoff_rigidity"]
            altitudes = graph.graph["metadata_altitude"]

            ax = axes[i]
            i += 1

            if fixed_node_size is None:
                node_sizes = np.array([altitudes[n] for n in graph.nodes()])
                node_sizes = altitude_normalizer.normalize(node_sizes)
                if altitude_normalizer != Normalizer.NONE:
                    node_sizes = 300 + (node_sizes - np.min(node_sizes)) * 700

                node_sizes = node_sizes.tolist()
            else:
                node_sizes = fixed_node_size

            pos = nx.spring_layout(graph, weight="weight", seed=42)
            nodes = list(graph.nodes())
            vals = np.array([cutoff.get(n, np.nan) for n in nodes], dtype=float)

            if np.all(np.isnan(vals)):
                nx.draw(
                    graph,
                    pos,
                    ax=ax,
                    with_labels=True,
                    node_size=node_sizes,
                    font_size=9,
                    font_color="black",
                    node_color="red",
                    edge_color="gray",
                )
            else:
                vmin = np.nanmin(vals)
                vmax = np.nanmax(vals)
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # type: ignore

                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=nodes,
                    node_color=vals,
                    cmap=cmap,
                    node_size=node_sizes,
                    ax=ax,
                )
                nx.draw_networkx_labels(graph, pos, font_size=9, ax=ax)
                nx.draw_networkx_edges(graph, pos, edge_color="gray", ax=ax)

                sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)  # type: ignore
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label("Cutoff rigidity")

            ax.set_title(
                (
                    f"Event Date: {event_date} -- Intensity: {intensity} -- "
                    f"Drop: {drop} -- Adj Method: {adj_method.title()}"
                )
            )

    if title:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout()
    plt.show()

    return fig
