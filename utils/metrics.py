import random
from typing import List, Optional, Set, Tuple

import networkx as nx
import numpy as np


def box_covering_cbb(
    graph: nx.Graph,
    lb: int,
    weight: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Set], int]:
    """
    Compact-Box-Burning (CBB) box covering algorithm.

    Args:
        graph (nx.Graph): Input graph
        lb (int): Box diameter threshold
        weight (str | None): Edge weight attribute (None -> unweighted)
        seed (int | None): Random seed

    Returns:
        Tuple[List[Set], int]:
            List of boxes (set of nodes) and number of boxes

    Reference:
        Song, Chaoming & Gallos, Lazaros & Havlin, Shlomo & Makse, Hernan. (2007).
        How to calculate the fractal dimension of a complex network: The box covering algorithm.
        Journal of Statistical Mechanics: Theory and Experiment. 2007. 10.1088/1742-5468/2007/03/P03006.
    """
    rng = random.Random(seed)

    uncovered = set(graph.nodes())
    boxes = []

    # Precompute all-pairs shortest paths up to lB-1
    # (optimization: BFS/Dijkstra truncated)
    distances = {}
    for node in graph.nodes():
        d = nx.single_source_dijkstra_path_length(
            graph,
            node,
            cutoff=lb - 1,
            weight=weight,
        )
        distances[node] = d

    while uncovered:
        # Candidate set
        C = set(uncovered)
        box = set()

        while C:
            p = rng.choice(tuple(C))

            box.add(p)
            C.remove(p)

            # Remove nodes too far from p
            to_remove = set()
            for q in C:
                dist = distances[p].get(q, float("inf"))
                if dist >= lb:
                    to_remove.add(q)

            C -= to_remove

        boxes.append(box)
        uncovered -= box

    return boxes, len(boxes)


def graph_fractal_dimension(
    graph: nx.Graph,
    weight: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[float, List[float], List[float]]:
    thresholds = range(2, 12)
    counts = []

    for threshold in thresholds:
        _boxes, count = box_covering_cbb(graph, threshold, weight=weight, seed=seed)
        counts.append(count)
    
    # Calculate fractal dimension
    thresholds = np.array(thresholds)
    counts = np.array(counts)

    log_thresholds = np.log(thresholds)
    log_counts = np.log(counts)
    coef = np.polyfit(log_thresholds, log_counts, 1)
    fractal_dimension = -coef[0]
    return fractal_dimension, log_thresholds.tolist(), log_counts.tolist()


"""
Box Counting hecho por David Sierra

# Función para calcular la dimensión fractal (Método de Box-counting)
def calcular_dimension_fractal(imagen):
    # Convertir la imagen a escala de grises y binarizarla
    # imagen_gray = imagen.convert("L")
    pixels = np.array(imagen)
    pixels = (pixels > np.mean(pixels)).astype(int)  # Binarizar la imagen

    # Tamaños de los "cajas" que se utilizarán para el conteo
    sizes = []
    counts = []

    for size in range(2, min(pixels.shape) // 2):
        count = 0
        # Contar cuántas "cajas" de tamaño `size` contienen píxeles blancos
        for i in range(0, pixels.shape[0], size):
            for j in range(0, pixels.shape[1], size):
                if np.any(pixels[i : i + size, j : j + size] == 1):
                    count += 1
        sizes.append(size)
        counts.append(count)

    # Calcular la dimensión fractal
    sizes = np.array(sizes)
    counts = np.array(counts)

    # Calcular la pendiente de la regresión log-log
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    coef = np.polyfit(log_sizes, log_counts, 1)  # Regresión lineal
    dimension_fractal = -coef[0]  # La pendiente es la dimensión fractal
    return dimension_fractal, log_sizes, log_counts
"""
