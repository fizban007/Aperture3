#!/usr/bin/env python

import h5py
import matplotlib
matplotlib.use("Agg")
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import pytoml
from pathlib import Path
from multiprocessing import Pool

class Data:
    def __init__(self, conf):
        N1 = conf['Grid']['N'][0] // conf['Simulation']['downsample'] + conf['Grid']['guard'][0] * 2
        N2 = conf['Grid']['N'][1] // conf['Simulation']['downsample'] + conf['Grid']['guard'][1] * 2
        self.rho_e = np.zeros((N1, N2))
        self.rho_p = np.zeros((N1, N2))
        self.rho_i = np.zeros((N1, N2))
        self.pairs = np.zeros((N1, N2))
        self.E1 = np.zeros((N1, N2))
        self.E2 = np.zeros((N1, N2))
        self.E3 = np.zeros((N1, N2))
        self.B1 = np.zeros((N1, N2))
        self.B2 = np.zeros((N1, N2))
        self.B3 = np.zeros((N1, N2))
        self.j1 = np.zeros((N1, N2))
        self.j2 = np.zeros((N1, N2))
        self.j3 = np.zeros((N1, N2))
        self.flux = np.zeros((N1, N2))
        self.B = np.zeros((N1, N2))
        self.EdotB = np.zeros((N1, N2))

    def load(self, path):
        data = h5py.File(path, 'r', swmr=True)

        self.rho_e = data['Rho_e'].value
        self.rho_p = data['Rho_p'].value
        self.rho_i = data['Rho_i'].value
        self.pairs = data['pair_produced'].value
        self.E1 = data['E1'].value
        self.E2 = data['E2'].value
        self.E3 = data['E3'].value
        self.B1 = data['B1'].value + data['B_bg1'].value
        self.B2 = data['B2'].value + data['B_bg2'].value
        self.B3 = data['B3'].value + data['B_bg3'].value
        self.j1 = data['J1'].value
        self.j2 = data['J2'].value
        self.j3 = data['J3'].value
        self.flux = data['flux'].value
        self.B = np.sqrt(self.B1*self.B1 + self.B2*self.B2 + self.B3*self.B3)
        self.EdotB = (self.E1*self.B1 + self.E2*self.B2 + self.E3*self.B3)/self.B

        data.close()


class Plots:
    def __init__(self, data_dir, data_interval):
        self.data_dir = data_dir
        self.data_interval = data_interval
        # Define color map
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Wedge, Arc, Rectangle
        cdata = { 'red' :  [(0.0, 0.0, 0.0),
                            (0.5, 0.0, 0.0),
                            (0.55, 1.0, 1.0),
                            (1.0, 1.0, 1.0)],
                'green': [(0.0, 1.0, 1.0),
                            (0.45, 0.0, 0.0),
                            (0.55, 0.0, 0.0),
                            (1.0, 1.0, 1.0)],
                'blue' : [(0.0, 1.0, 1.0),
                            (0.45, 1.0, 1.0),
                            (0.5, 0.0, 0.0),
                            (1.0, 0.0, 0.0)]}

        hot_cold_cmap = LinearSegmentedColormap('hot_and_cold', cdata, N=1024, gamma=1.0)
        plt.register_cmap(cmap=hot_cold_cmap)
        
    def make_plot(self, step):
        plotfile = Path("plots/%06d.png" % (step // self.data_interval))
        if plotfile.exists():
            return
        conf_path = os.path.join(self.data_dir, 'config.toml')
        with open(conf_path, 'rb') as f:
            conf = pytoml.load(f)

        # print(conf)
        data = Data(conf)

        # Load mesh file
        mesh = h5py.File(os.path.join(self.data_dir, 'mesh.h5'), 'r', swmr=True)

        x1 = mesh['x1'].value
        x2 = mesh['x2'].value
        r2 = x1*x1+x2*x2
        r3 = r2**1.5

        mesh.close()

        data.load(os.path.join(self.data_dir, 'data%06d.h5' % step))

        # Actual plotting is here
        fig, axes = plt.subplots(2, 4)
        fig.set_size_inches(26, 22)

        color_flux = '#7cfc00'
        rho_max = 10000
        B_max = 500
        j_max = 4000
        pair_max = 200
        plot_rho_t = axes[0,0].pcolormesh(x1, x2, (data.rho_e + data.rho_p + data.rho_i)*r2, cmap='hot_and_cold', shading="gouraud", vmin=-rho_max, vmax=rho_max)
        plot_rho_e = axes[0,1].pcolormesh(x1, x2, data.rho_e*r2, cmap='hot_and_cold', shading="gouraud", vmin=-rho_max, vmax=rho_max)
        plot_rho_p = axes[0,2].pcolormesh(x1, x2, data.rho_p*r2, cmap='hot_and_cold', shading="gouraud", vmin=-rho_max, vmax=rho_max)
        plot_rho_i = axes[0,3].pcolormesh(x1, x2, data.rho_i*r2, cmap='hot_and_cold', shading="gouraud", vmin=-rho_max, vmax=rho_max)
        plot_EdotB = axes[1,0].pcolormesh(x1, x2, data.EdotB, cmap='hot_and_cold', shading="gouraud", vmin=-40, vmax=40)
        plot_pairs = axes[1,1].pcolormesh(x1, x2, data.pairs, cmap='inferno', shading="gouraud", vmin=0, vmax=pair_max)
        plot_jr = axes[1,2].pcolormesh(x1, x2, data.j1*r2, cmap='hot_and_cold', shading="gouraud", vmin=-j_max, vmax=j_max)
        plot_Bphi = axes[1,3].pcolormesh(x1, x2, data.B3, cmap='hot_and_cold', shading="gouraud", vmin=-B_max, vmax=B_max)

        lower_plots = [plot_EdotB, plot_pairs, plot_jr, plot_Bphi]

        y_lim = 10
        x_lim = 12
        flux_lower = 100
        flux_upper = 40000
        flux_num = 15
        clevel = np.linspace(flux_lower, flux_upper, flux_num)
        tick_size = 20
        contours = []
        for i in range(8):
            axes[i//4,i%4].set_xlim([0, x_lim])
            axes[i//4,i%4].set_ylim([-y_lim, y_lim])
            axes[i//4,i%4].set_aspect('equal', 'datalim')
            axes[i//4,i%4].tick_params(axis="both", labelsize=tick_size)
            axes[i//4,i%4].axvline(1.0/conf['omega'], linestyle='--', color='w')
            contours.append(axes[i//4,i%4].contour(x1, x2, data.flux, clevel, colors=color_flux, linewidths=0.6))


        pos0 = axes[0,3].get_position()
        cbar_ax0 = fig.add_axes([pos0.x0 + pos0.width + 0.01, pos0.y0, 0.01, pos0.height])
        cbar0 = fig.colorbar(plot_rho_i, cax = cbar_ax0)
        cbar_ax0.tick_params(labelsize=tick_size)

        title_size=40
        time_size=50
        axes[0,0].set_title("$\\rho_\mathrm{total}r^2$", fontsize=title_size)
        axes[0,1].set_title("$\\rho_e r^2$", fontsize=title_size)
        axes[0,2].set_title("$\\rho_p r^2$", fontsize=title_size)
        axes[0,3].set_title("$\\rho_i r^2$", fontsize=title_size)
        axes[1,0].set_title("$\mathbf{E}\cdot\mathbf{B}/B$", fontsize=title_size)
        axes[1,1].set_title("$e^\pm$ creation rate", fontsize=title_size)
        axes[1,2].set_title("$j_r r^2$", fontsize=title_size)
        axes[1,3].set_title("$B_\\phi$", fontsize=title_size)

        cbar_ax1 = []
        for i in range(4):
            pos1 = axes[1,i].get_position()
            cbar_ax1.append(fig.add_axes([pos1.x0, pos1.y0 - 0.03, pos1.width, 0.01]))
            cbar_ax1[i].tick_params(labelsize=tick_size)
            fig.colorbar(lower_plots[i], orientation='horizontal', cax=cbar_ax1[i])

        time_txt = plt.text(-0.1, 1.15, 'Time = {:.2f}'.format(step * conf['delta_t']),
            horizontalalignment='left',
            verticalalignment='center',
            transform = axes[0,0].transAxes,
            fontsize = time_size)
        # fig.savefig("pulsar_test.png")

        print("Plotting %d" % (step//self.data_interval))
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

plots = Plots(data_dir, data_interval)

agents = 7

steps_to_plot = list(range(0, steps, data_interval))
for step in steps_to_plot:
    plotfile = Path("plots/%06d.png" % (step // plots.data_interval))
    if plotfile.exists():
        steps_to_plot.remove(step)
chunksize = (len(steps_to_plot) + agents - 1) // agents

with Pool(processes=agents) as pool:
    pool.map(plots.make_plot, steps_to_plot, chunksize)
