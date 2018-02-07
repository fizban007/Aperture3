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

for n in range(250):
    print(n)
    # Change data directory!
    file_path = os.path.join(data_dir, "output%06d.h5" % (n * 20));
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

    ax1.plot(x_e, p_e, '.')
    ax1.plot(x_p, p_p, ".")
    ax1.set_ylabel('$p/mc$')
    ax1.set_xlabel('$x/\lambda_p$')
    ax1.set_xlim([conf['grid']['lower'], conf['grid']['lower'] + conf['grid']['size']])
    ax2 = ax1.twinx()
    guard = conf['grid']['guard']
    N = conf['grid']['N']
    xs = np.linspace(conf['grid']['lower'], conf['grid']['lower'] + conf['grid']['size'],
                     N - 2 * guard + 1)
    ax2.plot(xs, E1[guard-1:-guard])
    ax2.set_ylabel("$E$")

    fig.savefig("%06d.png" % n)
    plt.close(fig)
