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

data_dir = sys.argv[1]
if not os.path.isdir(data_dir):
    print("Given path is not a directory")
    exit(0)
if len(sys.argv) > 2:
    initial_step = int(sys.argv[2])
else:
    initial_step = 0

# conf = json.load(open(os.path.join(data_dir, "config.json")))
conf_path = os.path.join(data_dir, 'config.toml')
with open(conf_path, 'rb') as f:
    conf = pytoml.load(f)
jb = 1.0
N = conf['Grid']['N'][0]
xs = np.linspace(conf['Grid']['lower'][0], conf['Grid']['lower'][0] + conf['Grid']['lower'][0], N)

fig, axes = plt.subplots(1, 2)
fig.set_size_inches(18.5, 7.5)
ax2 = axes[0].twinx()

# fig.canvas.draw()

el, = axes[0].plot([], [], '.', markersize=1.0, color='blue', alpha=0.3)
po, = axes[0].plot([], [], '.', markersize=1.0, color='orange', alpha=0.3)
ph, = axes[0].plot([], [], '.', markersize=1.0, color='k', alpha=0.3)

axes[0].set_ylabel('$p/mc$')
axes[0].set_xlabel('$x/\lambda_p$')
print(conf['Grid']['lower'])
print(conf['Grid']['size'])
axes[0].set_xlim([conf['Grid']['lower'][0], conf['Grid']['lower'][0] + conf['Grid']['size'][0]])
thr = conf['gamma_thr'] * 1.8
# axes[0, 0].set_ylim([-1e5, 1e5])
# axes[0, 0].axvline(x=291.7/2, linestyle='--')

# guard = conf['grid']['guard']
# N = conf['grid']['N']
# xs = np.linspace(conf['grid']['lower'], conf['grid']['lower'] + conf['grid']['size'],
#                  N - 2 * guard + 1)
efield, = ax2.plot(xs, np.zeros(len(xs)), color='g', linewidth=0.6)
ax2.set_ylabel("$E$")
# ax2.set_ylim([-400, 400])

j, = axes[1].plot([], [])
rho, = axes[1].plot([], [])
axes[1].plot(xs, np.ones(len(xs)))
# axes[0,1].plot(xs, 2.0*np.ones(len(xs)))
# axes[0,1].axvline(x=291.7/2, linestyle='--')
# axes[0,1].set_ylim(bottom=0)
txt = axes[1].text(0.7,1.1,"$t = {}$".format(0.0),
                     size=20,transform = axes[1].transAxes)

# ve, = axes[1,0].plot([], [])
# vp, = axes[1,0].plot([], [])

# # axes[1,0].plot(xs,np.zeros(len(xs)))
# # axes[1,0].set_ylim([-1.2, 1.2])
# # axes[1,0].axvline(x=291.7/2, linestyle='--')
# a,b,hist_e = axes[1,0].hist(np.array([0]), bins=100, histtype=u'step')
# a,b,hist_p = axes[1,0].hist(np.array([0]), bins=100, histtype=u'step')
# a,b,hist_ph = axes[1,0].hist(np.array([0]), bins=100, histtype=u'step')
# axes[1,0].set_yscale('log')

# j, = axes[1,1].plot([],[])
# rho, = axes[1,1].plot([],[])
# axes[1,1].plot(xs,np.zeros(len(xs)))
# axes[1,1].plot(xs,1.0*np.ones(len(xs)),'--')
# # axes[1,1].plot(xs,(2.0*xs/1000.0 - 0.7)*5.0,'--')
# # axes[1,1].plot(xs,6.0 * np.cos(2.0 * np.pi * xs / conf["grid"]['size']))
# # axes[1,1].plot(xs,24.0 * (abs(0.5 - xs / conf["grid"]['size']) - 0.25))
# # axes[1,1].plot(xs,5.0 * (1.85 - 130.0 / (80.0 + 250.0 * xs / conf['grid']['size'])))
# axes[1,1].plot(xs,jb * (0.85 - 130.0 / (80.0 + 250.0 * (xs / conf['grid']['size'] - 0.03))))
# # axes[1,1].plot(xs,jb * 0.9 * (2.0 * xs / conf['grid']['size'] - 1.0))
# # axes[1,1].plot(xs, jb * np.arctan(5.0*(2.0*xs/conf['grid']['size']-1.2))*2.0/np.pi)
# # axes[1,1].plot(xs, jb * 1.75 * np.arctan(1.0*(2.0*xs/conf['grid']['size']-1.2))*2.0/np.pi)
# # axes[1,1].plot(xs,jb*(0.44 + 0.6*np.arctan(5.0*(2.0*xs/conf['grid']['size']-1.2))*2.0/np.pi))
# axes[1,1].set_ylim([-1.0, 1.5])
# axes[1,1].axvline(x=291.7/2, linestyle='--')

# axbg = fig.canvas.copy_from_bbox(ax1.bbox)
# images = []

data = Data(conf)

for step in range(initial_step, conf['Simulation']['max_steps'] // conf['Simulation']['data_interval']):
# for n in range(1000):
    print(step)
    # Change data directory!
    data.load(os.path.join(data_dir, 'data%06d.h5' % step))
    # file_path = os.path.join(data_dir, "data%06d.h5" % (n * conf['Simulation']['data_interval']))
    # file_path = os.path.join(data_dir, "output%06d.h5" % (n * 200))
    # with h5py.File(file_path, 'r', swmr=True) as f:
    #     E1 = f["E1"]
    #     J1 = f["J1"]
    #     rho_e = f["Rho_e"]
    #     rho_p = f["Rho_p"]
    el.set_data(data.x_e, data.p_e)
    po.set_data(data.x_p, data.p_p)
    # ax1.plot(x_p, p_p, ".", markersize=1.0)
    # if conf['trace_photons']:
    #     x_ph = np.array(f["Photons_x"])
    #     p_ph = np.array(f["Photons_p"])
    #     # ax1.plot(x_ph, p_ph, '.', markersize=1.0, color='k')
    #     ph.set_data(x_ph, p_ph)

    efield.set_data(xs, data.E1)

    # mult.set_data(xs, (rho_p - rho_e)/jb)
    # ve.set_data(xs, j_e/rho_e)#vp.set_data(xs, j_p / rho_p)
    # axes[1, 0].cla()
    # a, b, c = axes[1, 0].hist(np.log10(np.sqrt(p_e * p_e + 1) + 1), bins = 100, histtype = u'step')
    # a, b, c = axes[1, 0].hist(np.log10(np.sqrt(p_p * p_p + 1) + 1), bins = 100, histtype = u'step')
    # a, b, c = axes[1, 0].hist(np.log10(abs(p_ph) + 1), bins = 100, histtype = u'step')
    # axes[1, 0].set_yscale('log')
    # j.set_data(xs, (j_e + j_p) / jb)
    # rho.set_data(xs, (rho_e + rho_p) / jb)
    j.set_data(xs, data.J1)
    rho.set_data(xs, data.rho_e + data.rho_p)
    # axes[1].relim()
    axes[1].autoscale_view(True, True, True)
    # axes[0].relim()
    axes[0].autoscale_view(True, True, True)
    # ax2.relim()
    ax2.autoscale_view(True, True, True)
    txt.set_text("$t = {}$".format(step * conf["delta_t"] * conf['Simulation']['data_interval']))
    fig.savefig("1dplots/%06d.png" % step)
    plt.close(fig)

#axes[0, 1].set_ylim(bottom = 0)

#fig.canvas.restore_region(axbg)

#ax1.draw_artist(el)
#ax1.draw_artist(po)
#ax1.draw_artist(ph)
#ax2.draw_artist(efield)

#fig.canvas.blit(ax1.bbox)
#fig.canvas.flush_events()
#fig.canvas.draw()
#plt.draw()

