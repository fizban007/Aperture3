#!/usr/bin/env python

import h5py
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import sys
import os
import json
import numpy as np

data_dir = sys.argv[1]
if not os.path.isdir(data_dir):
    print("Given path is not a directory")
    exit(0)

for n in range(1000):
    print(n)
    # Change data directory!
    file_path = os.path.join(data_dir, "output%06d.h5" % (n * 100));
    f = h5py.File(file_path)
    conf = json.load(open(os.path.join(data_dir, "config.json")));
#     E1 = f["E1"]
#     J1 = f["J1"]
#     rho_e = f["Rho_e"]
#     rho_p = f["Rho_p"]
    x_e = f["Electrons_x"]
    p_e = f["Electrons_p"]
    x_p = f["Positrons_x"]
    p_p = f["Positrons_p"]
    E1 = f["E1"]
    fig, ax1 = plt.subplots()

    ax1.plot(x_e, p_e, '.', markersize=1.0)
    ax1.plot(x_p, p_p, ".", markersize=1.0)
    if conf['trace_photons']:
        x_ph = f["Photons_x"]
        p_ph = f["Photons_p"]
        ax1.plot(x_ph, p_ph, '.', markersize=1.0, color='k')
    ax1.set_ylabel('$p/mc$')
    ax1.set_xlabel('$x/\lambda_p$')
    ax1.set_xlim([conf['grid']['lower'], conf['grid']['lower'] + conf['grid']['size']])
    thr = conf['gamma_thr'] * 1.2
    ax1.set_ylim([-thr, thr])
    ax2 = ax1.twinx()
    guard = conf['grid']['guard']
    N = conf['grid']['N']
    xs = np.linspace(conf['grid']['lower'], conf['grid']['lower'] + conf['grid']['size'],
                     N - 2 * guard + 1)
    ax2.plot(xs, E1[guard-1:-guard],color='g',linewidth=0.6)
    ax2.set_ylabel("$E$")
    ax2.set_ylim([-30, 30])

    fig.savefig("%06d.png" % n)
    plt.close(fig)
