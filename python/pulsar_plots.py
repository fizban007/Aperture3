#!/usr/bin/env python

import h5py
import matplotlib

matplotlib.use("Agg")
matplotlib.rc("text", usetex=True)
matplotlib.rc("font", family="serif")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os
import numpy as np
import pytoml
from pathlib import Path
from multiprocessing import Pool
from datalib import Data


class Plots:
    def __init__(self, data_dir, data_interval, data):
        self.data_dir = data_dir
        self.data_interval = data_interval
        self.data = data
        # Define color map
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Wedge, Arc, Rectangle

        cdata = {
            "red": [
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (0.55, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ],
            "green": [
                (0.0, 1.0, 1.0),
                (0.45, 0.0, 0.0),
                (0.55, 0.0, 0.0),
                (1.0, 1.0, 1.0),
            ],
            "blue": [
                (0.0, 1.0, 1.0),
                (0.45, 1.0, 1.0),
                (0.5, 0.0, 0.0),
                (1.0, 0.0, 0.0),
            ],
        }

        hot_cold_cmap = LinearSegmentedColormap(
            "hot_and_cold", cdata, N=1024, gamma=1.0
        )
        plt.register_cmap(cmap=hot_cold_cmap)

    def make_plot(self, step):
        plotfile = Path("plots/%06d.png" % (step // self.data_interval))
        if plotfile.exists():
            return

        data = self.data
        # print(conf)
        conf = data.conf

        x1 = data.x1
        x2 = data.x2
        r2 = x1 * x1 + x2 * x2
        r3 = r2 ** 1.5

        data.load(step)

        # Actual plotting is here
        fig, axes = plt.subplots(2, 4)
        fig.set_size_inches(26, 22)

        color_flux = "#7cfc00"
        B_max = conf["B0"] * conf["omega"] * 0.1
        rho_max = 1.0
        j_max = rho_max
        rho_gj = 2.0 * conf["B0"] * conf["omega"]
        pair_max = 100
        #edotb_max = 0.005 * conf["B0"]
        edotb_max = 200.0
        plot_rho_t = axes[0, 0].pcolormesh(
            x1,
            x2,
            (data.rho_e + data.rho_p + data.rho_i) / rho_gj * r2,
            cmap="hot_and_cold",
            shading="gouraud",
            vmin=-rho_max,
            vmax=rho_max,
        )
        plot_rho_e = axes[0, 1].pcolormesh(
            x1,
            x2,
            data.rho_e / rho_gj * r2,
            cmap="hot_and_cold",
            shading="gouraud",
            vmin=-rho_max,
            vmax=rho_max,
        )
        plot_rho_p = axes[0, 2].pcolormesh(
            x1,
            x2,
            data.rho_p / rho_gj * r2,
            cmap="hot_and_cold",
            shading="gouraud",
            vmin=-rho_max,
            vmax=rho_max,
        )
        plot_rho_i = axes[0, 3].pcolormesh(
            x1,
            x2,
            data.rho_i / rho_gj * r2,
            cmap="hot_and_cold",
            shading="gouraud",
            vmin=-rho_max,
            vmax=rho_max,
        )
        plot_EdotB = axes[1, 0].pcolormesh(
            x1,
            x2,
            data.EdotB,
            cmap="hot_and_cold",
            shading="gouraud",
            vmin=-edotb_max,
            vmax=edotb_max,
        )
        plot_pairs = axes[1, 1].pcolormesh(
            x1,
            x2,
            0.001 + data.pairs,
            cmap="inferno",
            shading="gouraud",
            norm=LogNorm(vmin=0.001, vmax=pair_max),
        )
        plot_jr = axes[1, 2].pcolormesh(
            x1,
            x2,
            data.j1 / rho_gj * r2,
            cmap="hot_and_cold",
            shading="gouraud",
            vmin=-j_max,
            vmax=j_max,
        )
        plot_Bphi = axes[1, 3].pcolormesh(
            x1,
            x2,
            data.B3,
            cmap="hot_and_cold",
            shading="gouraud",
            vmin=-B_max,
            vmax=B_max,
        )

        lower_plots = [plot_EdotB, plot_pairs, plot_jr, plot_Bphi]

        y_lim = 10
        x_lim = 12
        flux_lower = 10
        flux_upper = 0.5 * conf["B0"]
        flux_num = 15
        clevel = np.linspace(flux_lower, flux_upper, flux_num)
        tick_size = 20
        pair_limit = conf["r_cutoff"] * conf["omega"]
        contours = []
        for i in range(8):
            axes[i // 4, i % 4].set_xlim([0, x_lim])
            axes[i // 4, i % 4].set_ylim([-y_lim, y_lim])
            if i == 5:
                axes[i // 4, i % 4].set_xlim([0, pair_limit*x_lim])
                axes[i // 4, i % 4].set_ylim([-pair_limit*y_lim, pair_limit*y_lim])
            axes[i // 4, i % 4].set_aspect("equal", "datalim")
            axes[i // 4, i % 4].tick_params(axis="both", labelsize=tick_size)
            axes[i // 4, i % 4].axvline(1.0 / conf["omega"], linestyle="--", color="w")
            if i // 4 == 0 or i % 4 > 1:
                contours.append(
                    axes[i // 4, i % 4].contour(
                        x1, x2, data.flux, clevel, colors=color_flux, linewidths=0.6
                    )
                )

        pos0 = axes[0, 3].get_position()
        cbar_ax0 = fig.add_axes(
            [pos0.x0 + pos0.width + 0.01, pos0.y0, 0.01, pos0.height]
        )
        cbar0 = fig.colorbar(plot_rho_i, cax=cbar_ax0)
        cbar_ax0.tick_params(labelsize=tick_size)

        title_size = 40
        time_size = 50
        axes[0, 0].set_title("$\\rho_\mathrm{total}r^2$", fontsize=title_size)
        axes[0, 1].set_title("$\\rho_e r^2$", fontsize=title_size)
        axes[0, 2].set_title("$\\rho_p r^2$", fontsize=title_size)
        axes[0, 3].set_title("$\\rho_i r^2$", fontsize=title_size)
        axes[1, 0].set_title("$\mathbf{E}\cdot\mathbf{B}/B$", fontsize=title_size)
        axes[1, 1].set_title("$e^\pm$ creation rate", fontsize=title_size)
        axes[1, 2].set_title("$j_r r^2$", fontsize=title_size)
        axes[1, 3].set_title("$B_\\phi$", fontsize=title_size)

        cbar_ax1 = []
        for i in range(4):
            pos1 = axes[1, i].get_position()
            cbar_ax1.append(fig.add_axes([pos1.x0, pos1.y0 - 0.03, pos1.width, 0.01]))
            cbar_ax1[i].tick_params(labelsize=tick_size)
            fig.colorbar(lower_plots[i], orientation="horizontal", cax=cbar_ax1[i])

        time_txt = plt.text(
            -0.1,
            1.15,
            "Time = {:.2f}".format(step * conf["delta_t"]),
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes[0, 0].transAxes,
            fontsize=time_size,
        )
        # fig.savefig("pulsar_test.png")

        print("Plotting %d" % (step // self.data_interval))
        fig.savefig("plots/%06d.png" % (step // self.data_interval))
        plt.close(fig)


# Read parameters
data_dir = sys.argv[1]

if not os.path.isdir(data_dir):
    print("Given path is not a directory")
    exit(0)
if len(sys.argv) > 2:
    steps = int(sys.argv[2])
    data_interval = int(sys.argv[3])
else:
    steps = 10000
    data_interval = 200

data = Data(data_dir)
plots = Plots(data_dir, data_interval, data)

agents = 7

# steps_to_plot = list(range(0, steps, data_interval))
steps_to_plot = [n for n in data.steps if n < steps]
for step in steps_to_plot:
    if step > steps or step % data_interval != 0:
        continue
    plotfile = Path("plots/%06d.png" % (step // data_interval))
    if plotfile.exists():
        steps_to_plot.remove(step)
chunksize = (len(steps_to_plot) + agents - 1) // agents

with Pool(processes=agents) as pool:
    pool.map(plots.make_plot, steps_to_plot, chunksize)
