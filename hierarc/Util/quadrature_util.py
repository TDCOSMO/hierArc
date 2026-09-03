"""Quadrature rules used by the deterministic marginalisation.

The rules here all return ``(nodes, weights)`` such that ``sum(weights * f(nodes))``
approximates the expectation of ``f`` under the corresponding probability distribution,
i.e. the weights carry the density and integrate to one (up to truncation).
"""

__author__ = "mmillon"

import numpy as np
from scipy.stats import norm


def gauss_legendre_panels(edges, n_per_panel):
    """Composite Gauss-Legendre nodes on a set of consecutive intervals.

    :param edges: increasing array of panel boundaries, length n_panel + 1
    :param n_per_panel: number of Gauss-Legendre nodes inside each panel
    :return: (nodes, weights) of the plain integral over [edges[0], edges[-1]]
    """
    x, w = np.polynomial.legendre.leggauss(int(n_per_panel))
    edges = np.asarray(edges, dtype=float)
    lo, hi = edges[:-1, None], edges[1:, None]
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return (centre + half * x[None, :]).ravel(), (half * w[None, :]).ravel()


def truncated_normal_panels(edges, mu, sigma, n_per_panel, mass_tol=1e-8):
    """Composite Gauss-Legendre rule for a normal density truncated to the panel span.

    Panels carrying a negligible share of the probability mass are dropped, so the node
    count adapts to how much of the range the distribution actually occupies.

    :param edges: increasing panel boundaries; the first and last are the truncation
        bounds. Interior edges should sit wherever the integrand loses smoothness.
    :param mu: mean of the untruncated normal
    :param sigma: standard deviation of the untruncated normal
    :param n_per_panel: Gauss-Legendre nodes per retained panel
    :param mass_tol: drop panels holding less than this fraction of the total mass
    :return: (nodes, weights); the weights sum to one over the retained panels
    """
    edges = np.asarray(sorted(set(float(e) for e in edges)), dtype=float)
    if sigma <= 0:
        # a delta function: the only sensible rule is the point itself
        return np.array([float(mu)]), np.array([1.0])
    total = norm.cdf((edges[-1] - mu) / sigma) - norm.cdf((edges[0] - mu) / sigma)
    if total <= 0:
        raise ValueError(
            "the truncation interval [%s, %s] carries no probability mass for "
            "N(%s, %s)" % (edges[0], edges[-1], mu, sigma)
        )
    x, w = np.polynomial.legendre.leggauss(int(n_per_panel))
    nodes, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mass = norm.cdf((hi - mu) / sigma) - norm.cdf((lo - mu) / sigma)
        if mass / total < mass_tol:
            continue
        centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        xs = centre + half * x
        nodes.append(xs)
        weights.append(half * w * norm.pdf(xs, mu, sigma) / total)
    if not nodes:
        raise ValueError("every panel was dropped; check mu, sigma and the bounds")
    return np.concatenate(nodes), np.concatenate(weights)


def histogram_nodes(bin_edges, pdf_array, n_per_bin=2):
    """Quadrature for a piecewise-constant (histogram) density.

    ``PDFSampling`` draws by inverting a linearly interpolated CDF, which is exactly a
    uniform draw inside a bin chosen with probability proportional to the histogram. The
    matching quadrature is therefore a rule that is uniform inside each bin.

    :param bin_edges: bin edges, length len(pdf_array) + 1
    :param pdf_array: histogram heights (need not be normalised)
    :param n_per_bin: Gauss-Legendre nodes inside each bin
    :return: (nodes, weights); the weights sum to one
    """
    bin_edges = np.asarray(bin_edges, dtype=float)
    p = np.asarray(pdf_array, dtype=float)
    p = p / np.sum(p)
    x, w = np.polynomial.legendre.leggauss(int(n_per_bin))
    u, wu = 0.5 * (x + 1.0), 0.5 * w
    lo, hi = bin_edges[:-1, None], bin_edges[1:, None]
    nodes = (lo + u[None, :] * (hi - lo)).ravel()
    weights = (p[:, None] * wu[None, :]).ravel()
    return nodes, weights


def product_rule(nodes_a, weights_a, nodes_b, weights_b, combine):
    """Combine two 1d rules into a rule for a derived scalar variable.

    :param combine: callable(a, b) -> value of the derived variable, broadcast over the
        outer product of the two node sets
    :return: (nodes, weights) of the derived variable, with len = len(a) * len(b)
    """
    a = np.asarray(nodes_a, dtype=float)[:, None]
    b = np.asarray(nodes_b, dtype=float)[None, :]
    values = combine(a, b).ravel()
    weights = np.outer(np.asarray(weights_a, dtype=float),
                       np.asarray(weights_b, dtype=float)).ravel()
    return values, weights
