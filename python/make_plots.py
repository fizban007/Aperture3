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
if len(sys.argv) > 2:
    initial_step = int(sys.argv[2])
else:
    initial_step = 0

conf = json.load(open(os.path.join(data_dir, "config.json")))

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(18.5, 10.5)
ax2 = axes[0, 0].twinx()

# fig.canvas.draw()

el, = axes[0, 0].plot([], [], '.', markersize=1.0, color='blue', alpha=0.3)
po, = axes[0, 0].plot([], [], '.', markersize=1.0, color='orange', alpha=0.3)
ph, = axes[0, 0].plot([], [], '.', markersize=1.0, color='k', alpha=0.3)

axes[0, 0].set_ylabel('$p/mc$')
axes[0, 0].set_xlabel('$x/\lambda_p$')
axes[0, 0].set_xlim([conf['grid']['lower'], conf['grid']['lower'] + conf['grid']['size']])
thr = conf['gamma_thr'] * 1.8
# axes[0, 0].set_ylim([-1e5, 1e5])
# axes[0, 0].axvline(x=291.7/2, linestyle='--')

guard = conf['grid']['guard']
N = conf['grid']['N']
xs = np.linspace(conf['grid']['lower'], conf['grid']['lower'] + conf['grid']['size'],
                 N - 2 * guard + 1)
efield, = ax2.plot(xs, np.zeros(len(xs)), color='g', linewidth=0.6)
ax2.set_ylabel("$E$")
# ax2.set_ylim([-400, 400])

mult, = axes[0,1].plot([], [])
axes[0,1].plot(xs, np.ones(len(xs)))
axes[0,1].plot(xs, 2.0*np.ones(len(xs)))
# axes[0,1].axvline(x=291.7/2, linestyle='--')
# axes[0,1].set_ylim(bottom=0)

ve, = axes[1,0].plot([], [])
vp, = axes[1,0].plot([], [])

# axes[1,0].plot(xs,np.zeros(len(xs)))
# axes[1,0].set_ylim([-1.2, 1.2])
# axes[1,0].axvline(x=291.7/2, linestyle='--')
a,b,hist_e = axes[1,0].hist(np.array([0]), bins=100, histtype=u'step')
a,b,hist_p = axes[1,0].hist(np.array([0]), bins=100, histtype=u'step')
a,b,hist_ph = axes[1,0].hist(np.array([0]), bins=100, histtype=u'step')
axes[1,0].set_yscale('log')

j, = axes[1,1].plot([],[])
rho, = axes[1,1].plot([],[])
axes[1,1].plot(xs,np.zeros(len(xs)))
axes[1,1].plot(xs,1.0*np.ones(len(xs)),'--')
# axes[1,1].plot(xs,(2.0*xs/1000.0 - 0.7)*5.0,'--')
# axes[1,1].plot(xs,6.0 * np.cos(2.0 * np.pi * xs / conf["grid"]['size']))
# axes[1,1].plot(xs,24.0 * (abs(0.5 - xs / conf["grid"]['size']) - 0.25))
# axes[1,1].plot(xs,5.0 * (1.85 - 130.0 / (80.0 + 250.0 * xs / conf['grid']['size'])))
axes[1,1].plot(xs,1.0 * (0.85 - 130.0 / (80.0 + 250.0 * xs / conf['grid']['size'])))
axes[1,1].set_ylim([-1.0, 1.5])
# axes[1,1].axvline(x=291.7/2, linestyle='--')

# axbg = fig.canvas.copy_from_bbox(ax1.bbox)
# images = []

for n in range(initial_step, conf['N_steps']):
# for n in range(1000):
    print(n)
    # Change data directory!
    file_path = os.path.join(data_dir, "output%06d.h5" % (n * conf['data_interval']))
    # file_path = os.path.join(data_dir, "output%06d.h5" % (n * 200))
    with h5py.File(file_path, 'r', swmr=True) as f:
    #     E1 = f["E1"]
    #     J1 = f["J1"]
    #     rho_e = f["Rho_e"]
    #     rho_p = f["Rho_p"]
        x_e = np.array(f["Electrons_x"])
        p_e = np.array(f["Electrons_p"])
        x_p = np.array(f["Positrons_x"])
        p_p = np.array(f["Positrons_p"])
        E1 = np.array(f["E1"])[guard-1:-guard]
        j_e = np.array(f["J_e_avg"])[guard-1:-guard]
        j_p = np.array(f["J_p_avg"])[guard-1:-guard]
        rho_e = np.array(f["Rho_e_avg"])[guard-1:-guard]
        rho_p = np.array(f["Rho_p_avg"])[guard-1:-guard]

        el.set_data(x_e, p_e)
        po.set_data(x_p, p_p)
        # ax1.plot(x_p, p_p, ".", markersize=1.0)
        if conf['trace_photons']:
            x_ph = np.array(f["Photons_x"])
            p_ph = np.array(f["Photons_p"])
            # ax1.plot(x_ph, p_ph, '.', markersize=1.0, color='k')
            ph.set_data(x_ph, p_ph)

        efield.set_data(xs, E1)

        mult.set_data(xs, (rho_p - rho_e)/10)
        # ve.set_data(xs, j_e/rho_e)
        # vp.set_data(xs, j_p/rho_p)
        axes[1,0].cla()
        a, b, c = axes[1,0].hist(np.log10(np.sqrt(p_e*p_e + 1)+1), bins=100, histtype=u'step')
        a, b, c = axes[1,0].hist(np.log10(np.sqrt(p_p*p_p + 1)+1), bins=100, histtype=u'step')
        a, b, c = axes[1,0].hist(np.log10(abs(p_ph)+1), bins=100, histtype=u'step')
        axes[1,0].set_yscale('log')
        j.set_data(xs, (j_e + j_p)/10)
        rho.set_data(xs, (rho_e + rho_p)/10)
        axes[0,1].relim()
        axes[0,1].autoscale_view(True,True,True)
        axes[0,0].relim()
        axes[0,0].autoscale_view(True,True,True)
        ax2.relim()
        ax2.autoscale_view(True,True,True)
        # axes[0,1].set_ylim(bottom=0)

        # fig.canvas.restore_region(axbg)

        # ax1.draw_artist(el)
        # ax1.draw_artist(po)
        # ax1.draw_artist(ph)
        # ax2.draw_artist(efield)

        # fig.canvas.blit(ax1.bbox)
        # fig.canvas.flush_events()
        # fig.canvas.draw()
        # plt.draw()

        fig.savefig("%06d.png" % n)
        plt.close(fig)
