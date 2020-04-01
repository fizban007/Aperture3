#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Wedge, Arc, Rectangle
from matplotlib import cm
import sys
import os
import numpy as np
from multiprocessing import Pool

from datalib_logsph import Data

cdata = {
    "red": [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.55, 1.0, 1.0), (1.0, 1.0, 1.0),],
    "green": [(0.0, 1.0, 1.0), (0.45, 0.0, 0.0), (0.55, 0.0, 0.0), (1.0, 1.0, 1.0),],
    "blue": [(0.0, 1.0, 1.0), (0.45, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),],
}

hot_cold_cmap = LinearSegmentedColormap("hot_and_cold", cdata, N=1024, gamma=1.0)
plt.register_cmap(cmap=hot_cold_cmap)
matplotlib.rc("text", usetex=True)
matplotlib.rc("font", family="serif")

data = Data(sys.argv[1])


def make_plot(n):
    data = Data(sys.argv[1])
    conf = data._conf
    print(conf["delta_t"] * n)
    print("Working on", n)
    data.load_fld(n)
    JdotB = (data.J1 * data.B1 + data.J2 * data.B2 + data.J3 * data.B3) / (
        data.B * data.B
    )
    mult = (abs(data.Rho_e) + abs(data.Rho_p) + abs(data.Rho_i)) / data.J

    fig, axes = plt.subplots(2, 3)
    fig.set_size_inches(28.5, 18.5)

    ticksize = 25
    titlesize = 35
    pc0 = axes[0, 0].pcolormesh(
        data.x1,
        data.x2,
        data.Rho_e * data.rv * data.rv,
        cmap=hot_cold_cmap,
        vmin=-15000,
        vmax=15000,
        shading="gouraud",
    )
    pc1 = axes[0, 1].pcolormesh(
        data.x1,
        data.x2,
        data.Rho_i * data.rv * data.rv,
        cmap=hot_cold_cmap,
        vmin=-15000,
        vmax=15000,
        shading="gouraud",
    )
    pc2 = axes[0, 2].pcolormesh(
        data.x1,
        data.x2,
        data.EdotB_avg,
        cmap=hot_cold_cmap,
        vmin=-100,
        vmax=100,
        shading="gouraud",
    )
    pc3 = axes[1, 0].pcolormesh(
        data.x1,
        data.x2,
        data.Rho_p * data.rv * data.rv,
        cmap=hot_cold_cmap,
        vmin=-15000,
        vmax=15000,
        shading="gouraud",
    )
    pc4 = axes[1, 1].pcolormesh(
        data.x1,
        data.x2,
        data.pair_produced,
        cmap=cm.hot,
        vmin=0,
        vmax=1,
        shading="gouraud",
    )
    pc5 = axes[1, 2].pcolormesh(
        data.x1, data.x2, JdotB, cmap=hot_cold_cmap, vmin=-1, vmax=1, shading="gouraud"
    )
    pcm = [pc0, pc1, pc2, pc3, pc4, pc5]

    i = 0
    for ax in axes.flatten():
        ax.set_aspect("equal")
        ax.set_xlim(0, 10)
        ax.set_ylim(-6, 6)
        ax.tick_params(axis="both", labelsize=ticksize)
        cb = fig.colorbar(pcm[i], ax=ax)
        cb.ax.tick_params(labelsize=ticksize)
        i += 1
    axes[0, 0].set_title("$\\rho_e r^2$", fontsize=titlesize)
    axes[0, 1].set_title("$\\rho_i r^2$", fontsize=titlesize)
    axes[0, 2].set_title("$E\cdot B/B$", fontsize=titlesize)
    axes[1, 0].set_title("$\\rho_p r^2$", fontsize=titlesize)
    axes[1, 1].set_title("Pair Production Rate", fontsize=titlesize)
    axes[1, 2].set_title("$J\cdot B/B^2$", fontsize=titlesize)
    axes[0, 2].text(
        7.0,
        7.5,
        "Time = %.2f" % (conf["delta_t"] * n * conf["Simulation"]["data_interval"]),
        fontsize=titlesize,
    )
    print("saving plot to plots/%05d.png" % n)
    fig.savefig("plots/%05d.png" % n)
    plt.close(fig)


steps_to_plot = data.fld_steps
agents = 7

with Pool(processes=agents) as pool:
    pool.map(make_plot, steps_to_plot)
