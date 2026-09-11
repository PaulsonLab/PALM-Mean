#!/usr/bin/env python3
"""Animate global minimization of a FIXED 1-D RBF GP posterior mean.

Install: python -m pip install numpy scipy matplotlib pillow
Run:     python animate_palm_mean_1d.py
         python animate_palm_mean_1d.py --save-frames --segments 3 --tol 0.001
Custom:  python animate_palm_mean_1d.py --x X.npy --y Y.npy --lengthscale 0.12

Based on https://github.com/PaulsonLab/PALM-Mean (Analytical_LB.py,
PWL_Custom_LB.py, UpperBound.py, BranchBound1.py). This is an independent
1-D adaptation, not a call to the original solver. Positive RBF terms use
the maximum of tangent lines in squared-distance space; negative terms use
secant interpolation. Small terms use analytical interval minima. In 1-D,
the resulting bound is piecewise quadratic in x, so all breakpoints and
stationary points can be enumerated instead of using Gurobi's MIQCP solver.
The branching policy is best-lower-bound selection and midpoint bisection.
Default preprocessing is disabled to show the branching process clearly.
Only incumbent evaluations and local search give upper bounds; plotting
grids play no role in bounding, pruning, or convergence.

Bounds are computed in float64 with a small downward numerical guard, not
formally certified interval arithmetic. GP hyperparameters stay fixed.
Training data must be scalar-output, with no implicit standardization.
--noise is observation noise VARIANCE; --signal-variance is kernel variance.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


class GPMean:
    def __init__(self, x, y, lengthscale=0.09, variance=1., noise=1e-5, mean=0.):
        self.x, self.y = np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)
        self.ell, self.variance, self.offset = lengthscale, variance, mean
        k = variance * np.exp(-0.5 * ((self.x[:, None]-self.x)/lengthscale)**2)
        self.c = variance * cho_solve(cho_factor(k + noise*np.eye(len(x))), self.y-mean)

    def __call__(self, x):
        z = (np.asarray(x)[..., None]-self.x)/self.ell
        return self.offset + np.exp(-0.5*z*z) @ self.c

    def gradient(self, x):
        dx = (np.asarray(x)[..., None]-self.x)
        return (np.exp(-0.5*(dx/self.ell)**2)*(-dx/self.ell**2)) @ self.c


class Relaxation:
    """A pointwise lower estimator on [a,b], minimized without a grid."""
    def __init__(self, gp, a, b, segments, threshold):
        self.gp, self.constant, self.terms = gp, gp.offset, []
        breaks = [a, b]
        for center, coefficient in zip(gp.x, gp.c):
            near = np.clip(center, a, b)
            lo = ((near-center)/gp.ell)**2
            hi = (max(abs(a-center), abs(b-center))/gp.ell)**2
            if abs(coefficient)*np.exp(-lo/2) < threshold or hi-lo < 1e-14:
                self.constant += coefficient*np.exp(-0.5*(hi if coefficient >= 0 else lo))
                continue
            knots = np.linspace(lo, hi, segments+1)
            values = coefficient*np.exp(-knots/2)
            if coefficient > 0:
                slopes = -values/2
                intercepts = values-slopes*knots
                # Adjacent tangent crossings delimit the maximum envelope.
                denom = slopes[:-1]-slopes[1:]
                cross = np.divide(intercepts[1:]-intercepts[:-1], denom,
                                  out=(knots[:-1]+knots[1:])/2, where=denom != 0)
                distance_breaks = cross[(cross > lo) & (cross < hi)]
            else:
                slopes = np.diff(values)/np.diff(knots)
                intercepts = values[:-1]-slopes*knots[:-1]
                distance_breaks = knots[1:-1]
            self.terms.append((center, coefficient, knots, slopes, intercepts))
            for d in distance_breaks:
                radius = gp.ell*np.sqrt(max(0., d))
                breaks.extend(t for t in (center-radius, center+radius) if a < t < b)
        self.breaks = np.unique(breaks)

    def line(self, term, x):
        center, coefficient, knots, slopes, intercepts = term
        d = ((x-center)/self.gp.ell)**2
        if coefficient > 0:
            j = np.argmax(slopes*d+intercepts)
        else:
            j = np.clip(np.searchsorted(knots, d, side='right')-1, 0, len(slopes)-1)
        return slopes[j], intercepts[j]

    def __call__(self, x):
        out = np.full_like(np.asarray(x, dtype=float), self.constant)
        for term in self.terms:
            center, coefficient, knots, slopes, intercepts = term
            d = ((np.asarray(x)-center)/self.gp.ell)**2
            if coefficient > 0:
                out += np.max(d[..., None]*slopes+intercepts, axis=-1)
            else:
                out += np.interp(d, knots, coefficient*np.exp(-knots/2))
        return out

    def minimum(self):
        candidates = list(self.breaks)
        for left, right in zip(self.breaks[:-1], self.breaks[1:]):
            quadratic, linear = 0., 0.
            for term in self.terms:
                slope, _ = self.line(term, (left+right)/2)
                quadratic += slope/self.gp.ell**2
                linear -= 2*slope*term[0]/self.gp.ell**2
            if quadratic > 0:
                stationary = -linear/(2*quadratic)
                if left < stationary < right:
                    candidates.append(stationary)
        values = self(np.asarray(candidates))
        idx = int(np.argmin(values))
        guard = 1e-10*(1+np.abs(self.gp.c).sum())
        return float(values[idx]-guard), float(candidates[idx])


@dataclass
class Node:
    a: float
    b: float
    lb: float
    relaxation: Relaxation


def search(gp, args):
    active, pruned, history, evaluations = [], [], [], []
    best_x, ub = args.lower, float(gp(args.lower))

    def make_node(a, b):
        nonlocal best_x, ub
        relaxation = Relaxation(gp, a, b, args.segments, args.threshold)
        lb, relaxed_x = relaxation.minimum()
        # Deterministic local starts; feasible results always provide UBs.
        candidates = [a, b, (a+b)/2, relaxed_x]
        for start in ((a+b)/2, relaxed_x):
            result = minimize(lambda z: float(gp(z[0])), [start],
                              jac=lambda z: np.array([gp.gradient(z[0])]),
                              method='L-BFGS-B', bounds=[(a, b)])
            candidates.append(float(np.clip(result.x[0], a, b)))
        values = gp(np.array(candidates))
        evaluations.extend(zip(candidates, values.tolist()))
        j = int(np.argmin(values))
        if values[j] < ub:
            best_x, ub = candidates[j], float(values[j])
        return Node(a, b, lb, relaxation)

    def snapshot(message, selected=None):
        # Prune only LB >= incumbent. Every removed interval is then known
        # to be no better than the current (or any future) incumbent.
        global_lb = min([ub] + [n.lb for n in active])
        history.append(dict(active=active.copy(), pruned=pruned.copy(),
                            ub=ub, lb=global_lb, x=best_x, message=message,
                            selected=selected, evaluations=evaluations.copy()))

    edges = np.linspace(args.lower, args.upper, args.prepart+1)
    active.extend(make_node(a, b) for a, b in zip(edges[:-1], edges[1:]))
    snapshot('Initialize interval bounds and local search')
    for _ in range(args.max_steps):
        if ub-min([ub]+[n.lb for n in active]) <= args.tol:
            snapshot('Converged: global bound gap meets tolerance')
            break
        node = min(active, key=lambda n: n.lb)
        snapshot('Select interval with the smallest lower bound', node)
        active.remove(node)
        mid = (node.a+node.b)/2
        if mid == node.a or mid == node.b:
            active.append(node)
            snapshot('Stopped: floating-point interval resolution reached')
            break
        active.extend([make_node(node.a, mid), make_node(mid, node.b)])
        snapshot('Bisect interval and recompute bounds', node)
        rejected = [n for n in active if n.lb >= ub]
        if rejected:
            pruned.extend(rejected)
            active[:] = [n for n in active if n.lb < ub]
            snapshot(f'Prune {len(rejected)} interval(s): lower bound >= incumbent')
    else:
        snapshot('Stopped at step limit; inspect the remaining bound gap')
    return history


def animate(gp, history, args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(args.lower, args.upper, 1800)
    curve = gp(grid)
    ymax = max(curve.max(), gp.y.max())
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=args.dpi)
    fig.subplots_adjust(left=.10, right=.97, bottom=.20, top=.80)
    legend = [Line2D([], [], color='#243746', lw=2.5, label='GP posterior mean'),
              Line2D([], [], color='#167d9a', lw=1.8, ls='--', label='Lower estimator'),
              Line2D([], [], color='#d56632', marker='*', ls='', markersize=12, label='Incumbent'),
              Patch(facecolor='#dceef2', label='Active interval'),
              Patch(facecolor='#e5e7eb', label='Pruned interval')]
    fig.legend(handles=legend, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .015))

    def draw(i):
        ax.clear()
        h = history[i]
        ymin = min(curve.min(), gp.y.min(), h['lb'])
        padding = .15*max(ymax-ymin, 1.)
        for n in h['pruned']:
            ax.axvspan(n.a, n.b, color='#e5e7eb', alpha=.7)
        for n in h['active']:
            ax.axvspan(n.a, n.b, color='#dceef2', alpha=.6)
            xx = np.linspace(n.a, n.b, 160)
            ax.plot(xx, n.relaxation(xx), '--', color='#167d9a', lw=1.3)
            ax.hlines(n.lb, n.a, n.b, color='#167d9a', lw=2.3)
            ax.axvline(n.a, color='white', lw=1)
        if h['selected'] is not None:
            n = h['selected']
            ax.axvspan(n.a, n.b, facecolor='#efb35b', alpha=.20)
        ax.plot(grid, curve, color='#243746', lw=2.7, zorder=4)
        ax.scatter(gp.x, gp.y, s=26, facecolors='white', edgecolors='#243746', zorder=5)
        ev = np.asarray(h['evaluations'])
        ax.scatter(ev[:, 0], ev[:, 1], s=10, color='#d56632', alpha=.4, zorder=5)
        ax.axhline(h['ub'], color='#d56632', lw=1, ls=':')
        ax.axhline(h['lb'], color='#167d9a', lw=1, ls=':')
        ax.scatter([h['x']], [h['ub']], marker='*', s=190, color='#d56632',
                   edgecolor='white', linewidth=.7, zorder=7)
        ax.set(xlim=(args.lower, args.upper), ylim=(ymin-padding, ymax+padding),
               xlabel='x', ylabel='Posterior mean / bounds')
        ax.set_title(h['message'], fontsize=12, pad=13)
        fig.suptitle('Global minimization of a GP posterior mean', y=.97, fontsize=17, weight='bold')
        gap = max(0., h['ub']-h['lb'])
        ax.text(0, 1.16, f"UB = {h['ub']:.5f}    LB = {h['lb']:.5f}    Gap = {gap:.2e}"
                f"    Active intervals: {len(h['active'])}", transform=ax.transAxes, fontsize=10)
        ax.text(.99, .03, f"Frame {i+1}/{len(history)}   |   x best = {h['x']:.5f}",
                ha='right', transform=ax.transAxes, fontsize=9, color='#64748b')
        return []

    # Hold the last frame for two seconds, without changing the search log.
    frames = list(range(len(history))) + [len(history)-1]*(2*args.fps)
    animation = FuncAnimation(fig, draw, frames=frames, interval=1000/args.fps, blit=False)
    animation.save(out/'palm_mean_1d.gif', writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    if args.save_frames:
        (out/'frames').mkdir(exist_ok=True)
        for i in range(len(history)):
            draw(i)
            fig.savefig(out/'frames'/f'frame_{i:04d}.png', dpi=args.dpi)
    draw(len(history)-1)
    fig.savefig(out/'palm_mean_1d_final.png', dpi=args.dpi)
    plt.close(fig)
    trace = np.array([[i, h['x'], h['ub'], h['lb'], h['ub']-h['lb'], len(h['active'])]
                      for i, h in enumerate(history)])
    np.savetxt(out/'bounds.csv', trace, delimiter=',', comments='',
               header='frame,best_x,upper_bound,lower_bound,gap,active_intervals')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--x', type=Path, help='Optional X.npy, shape (n,) or (n,1)')
    p.add_argument('--y', type=Path, help='Optional Y.npy, shape (n,) or (n,1)')
    p.add_argument('--lengthscale', type=float, default=.09)
    p.add_argument('--signal-variance', type=float, default=1.)
    p.add_argument('--noise', type=float, default=1e-5)
    p.add_argument('--mean', type=float, default=0.)
    p.add_argument('--lower', type=float, default=0.)
    p.add_argument('--upper', type=float, default=1.)
    p.add_argument('--segments', type=int, default=3)
    p.add_argument('--threshold', type=float, default=.01)
    p.add_argument('--prepart', type=int, default=1)
    p.add_argument('--tol', type=float, default=1e-3)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--fps', type=int, default=2)
    p.add_argument('--dpi', type=int, default=110)
    p.add_argument('--output', default='palm_animation')
    p.add_argument('--save-frames', action='store_true')
    args = p.parse_args()
    if bool(args.x) != bool(args.y):
        p.error('Provide both --x and --y.')
    if (args.lower >= args.upper or min(args.lengthscale, args.signal_variance, args.noise,
            args.tol, args.segments, args.prepart, args.max_steps, args.fps, args.dpi) <= 0
            or args.threshold < 0):
        p.error('Require increasing bounds, positive settings and nonnegative threshold.')
    if args.x:
        x, y = np.load(args.x), np.load(args.y)
        if any(v.ndim not in (1, 2) or (v.ndim == 2 and v.shape[1] != 1) for v in (x, y)):
            p.error('X and Y must have shape (n,) or (n,1).')
        x, y = x.ravel(), y.ravel()
    else:
        x = args.lower + (args.upper-args.lower)*np.array([.03, .14, .27, .40, .53, .65, .78, .90, .98])
        y = np.array([.35, -.70, .45, -.90, .55, -.45, .35, -.80, .20])
    if len(x) == 0 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        p.error('Require finite, nonempty X and Y of equal length.')
    gp = GPMean(x, y, args.lengthscale, args.signal_variance, args.noise, args.mean)
    history = search(gp, args)
    h = history[-1]
    print(f"{h['message']}\nx* = {h['x']:.8f}\nUB = {h['ub']:.8f}\n"
          f"LB = {h['lb']:.8f}\nGap = {h['ub']-h['lb']:.3e}\nFrames = {len(history)}")
    animate(gp, history, args)
    print(f'Outputs saved in {Path(args.output).resolve()}')


if __name__ == '__main__':
    main()
