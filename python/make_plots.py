#!/usr/bin/env python

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os
import json
import numpy as np
from matplotlib.animation import ArtistAnimation

data_dir = sys.argv[1]
if not os.path.isdir(data_dir):
    print("Given path is not a directory")
    exit(0)

conf = json.load(open(os.path.join(data_dir, "config.json")))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

fig.canvas.draw()

el, = ax1.plot([], [], '.', markersize=1.0, color='blue')
po, = ax1.plot([], [], '.', markersize=1.0, color='orange')
ph, = ax1.plot([], [], '.', markersize=1.0, color='k')

ax1.set_ylabel('$p/mc$')
ax1.set_xlabel('$x/\lambda_p$')
ax1.set_xlim([conf['grid']['lower'], conf['grid']['lower'] + conf['grid']['size']])
thr = conf['gamma_thr'] * 1.8
ax1.set_ylim([-thr, thr])

guard = conf['grid']['guard']
N = conf['grid']['N']
xs = np.linspace(conf['grid']['lower'], conf['grid']['lower'] + conf['grid']['size'],
                 N - 2 * guard + 1)
efield, = ax2.plot(xs, np.zeros(len(xs)), color='g', linewidth=0.6)
ax2.set_ylabel("$E$")
ax2.set_ylim([-50, 50])

axbg = fig.canvas.copy_from_bbox(ax1.bbox)
images = []

for n in range(1000):
    print(n)
    # Change data directory!
    file_path = os.path.join(data_dir, "output%06d.h5" % (n * 200))
    with h5py.File(file_path, 'r', swmr=True) as f:
    #     E1 = f["E1"]
    #     J1 = f["J1"]
    #     rho_e = f["Rho_e"]
    #     rho_p = f["Rho_p"]
        x_e = f["Electrons_x"]
        p_e = f["Electrons_p"]
        x_p = f["Positrons_x"]
        p_p = f["Positrons_p"]
        E1 = f["E1"]

        el.set_data(x_e, p_e)
        po.set_data(x_p, p_p)
        # ax1.plot(x_p, p_p, ".", markersize=1.0)
        if conf['trace_photons']:
            x_ph = f["Photons_x"]
            p_ph = f["Photons_p"]
            # ax1.plot(x_ph, p_ph, '.', markersize=1.0, color='k')
            ph.set_data(x_ph, p_ph)

        efield.set_data(xs, E1[guard-1:-guard])

        fig.canvas.restore_region(axbg)

        ax1.draw_artist(el)
        ax1.draw_artist(po)
        ax1.draw_artist(ph)
        ax2.draw_artist(efield)

        fig.canvas.blit(ax1.bbox)
        fig.canvas.flush_events()

        fig.savefig("%06d.png" % n)
        plt.close(fig)
