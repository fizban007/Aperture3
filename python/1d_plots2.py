#!/usr/bin/env python

import h5py
import matplotlib
matplotlib.use("Agg")
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
import matplotlib.pyplot as plt
import sys
import os
import pytoml
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from matplotlib.animation import ArtistAnimation

class Data:
    def __init__(self, conf):
        N = conf['Grid']['N'][0]
        self.rho_e = np.zeros(N)
        self.rho_p = np.zeros(N)
        self.E1 = np.zeros(N)
        self.J1 = np.zeros(N)
        self.x_e = np.array([])
        self.x_p = np.array([])
        self.p_e = np.array([])
        self.p_p = np.array([])

    def load(self, path):
        data = h5py.File(path, 'r', swmr=True)
        self.rho_e = np.array(data['Rho_e'])
        self.rho_p = np.array(data['Rho_p'])
        self.E1 = np.array(data['E1'])
        self.J1 = np.array(data['J1'])
        self.x_e = np.array(data['Electron_x'])
        self.p_e = np.array(data['Electron_p'])
        self.x_p = np.array(data['Positron_x'])
        self.p_p = np.array(data['Positron_p'])
        data.close()

data_dir = '/home/alex/storage/Data/Aperture3/1D_test/'
if not os.path.isdir(data_dir):
    print("Given path is not a directory")
    exit(0)
    
initial_step = 0

# conf = json.load(open(os.path.join(data_dir, "config.json")))
conf_path = os.path.join(data_dir, 'config.toml')
with open(conf_path, 'rb') as f:
    conf = pytoml.load(f)
jb = 1.0
N = conf['Grid']['N'][0] + 2 * conf['Grid']['guard'][0]
xs = np.linspace(conf['Grid']['lower'][0], conf['Grid']['lower'][0] + conf['Grid']['size'][0], N)

data = Data(conf)

for n in range(initial_step, conf['Simulation']['max_steps'] // conf['Simulation']['data_interval']):
    step = n * conf['Simulation']['data_interval']
    print(step)
    # Change data directory!
    data.load(os.path.join(data_dir, 'data%06d.h5' % step))
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(18.5, 7.5)
    ax2 = axes[0].twinx()

    el, = axes[0].plot(data.x_e, data.p_e, '.', markersize=1.0, color='blue', alpha=0.3)
    po, = axes[0].plot(data.x_p, data.p_p, '.', markersize=1.0, color='orange', alpha=0.3)
#     ph, = axes[0].plot([], [], '.', markersize=1.0, color='k', alpha=0.3)

    axes[0].set_ylabel('$p/mc$')
    axes[0].set_xlabel('$x/\lambda_p$')
#     print(conf['Grid']['lower'])
#     print(conf['Grid']['size'])
    axes[0].set_xlim([conf['Grid']['lower'][0], conf['Grid']['lower'][0] + conf['Grid']['size'][0]])
    thr = conf['gamma_thr'] * 1.8

    efield, = ax2.plot(xs, data.E1, color='g', linewidth=0.6)
    ax2.set_ylabel("$E$")
    # ax2.set_ylim([-400, 400])

    j, = axes[1].plot(xs, data.J1)
    rho, = axes[1].plot(xs, data.rho_e + data.rho_p)
    axes[1].plot(xs, np.ones(len(xs)))
    # axes[0,1].plot(xs, 2.0*np.ones(len(xs)))
    # axes[0,1].axvline(x=291.7/2, linestyle='--')
    # axes[0,1].set_ylim(bottom=0)
    txt = axes[1].text(0.7,1.1,"$t = {}$".format(0.0),
                         size=20,transform = axes[1].transAxes)
    txt.set_text("$t = {}$".format(step * conf["delta_t"]))
    fig.savefig("1dplots/%06d.png" % n)
    plt.close(fig)
