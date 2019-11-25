#!/usr/bin/env python

import matplotlib

matplotlib.use("Agg")
matplotlib.rc("text", usetex=True)
matplotlib.rc("font", family="serif")

import sys
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool
from datalib import Data
import numpy as np
import h5py
import glob
import pytoml
import re


rp = 1.14107
rm = 0.858933
a = 0.99

data_dir = sys.argv[1]
meshfile = h5py.File(Path(data_dir) / "mesh.h5")

xs = meshfile["x1"][()]

meshfile.close()

coeffile = h5py.File(Path(data_dir) / "../coef.h5")

rl_inner = coeffile["rL_inner"][()]
rl_outer = coeffile["rL_outer"][()]
rs = coeffile["r"][()]
rho0 = coeffile["rho"][()]
j0 = coeffile["ju1"][()]
r_null = rs[np.argmin(abs(rho0))]

coeffile.close()

with open(Path(data_dir) / "config.toml") as f:
    conf = pytoml.load(f)

def xi(r):
    return np.log((r - rp) / (r - rm)) / (rp - rm)


def rxi(xi):
    exp_xi = np.exp(xi * (rp - rm))
    return (rp - rm * exp_xi) / (1.0 - exp_xi)


def Delta_r(r):
    return r * r - 2.0 * r + a * a

label_size=30
tick_size=20

def make_plot(num):
    print("Working on", num)
    datafile = h5py.File(Path(data_dir) / ("data%06d.h5" % num))
    E1 = np.array(datafile["E1"])
    J1 = np.array(datafile["J1"])
    rho_e = np.array(datafile["Rho_e"])
    rho_p = np.array(datafile["Rho_p"])
    x_e = np.array(datafile["Electron_x"])
    p_e = np.array(datafile["Electron_p"])
    x_p = np.array(datafile["Positron_x"])
    p_p = np.array(datafile["Positron_p"])
    x_ph = np.array(datafile["photons_x"])
    p_ph = np.array(datafile["photons_p"])
    datafile.close()

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(18, 8)
    ax2 = axes[0].twinx()

    # el, = axes[0].plot(x_e, p_e/Delta_r(rxi(x_e)), '.', markersize=1.0, color='blue', alpha=0.3)
    # po, = axes[0].plot(x_p, p_p/Delta_r(rxi(x_p)), '.', markersize=1.0, color='orange', alpha=0.3)
    # ph, = axes[0].plot(x_ph, p_ph/Delta_r(rxi(x_ph)), '.', markersize=1.0, color='k', alpha=0.3)
    el, = axes[0].plot(rxi(x_e), p_e/Delta_r(rxi(x_e)), '.', markersize=1.0, color='blue', alpha=0.3)
    po, = axes[0].plot(rxi(x_p), p_p/Delta_r(rxi(x_p)), '.', markersize=1.0, color='orange', alpha=0.3)
    ph, = axes[0].plot(rxi(x_ph), p_ph/Delta_r(rxi(x_ph)), '.', markersize=1.0, color='k', alpha=0.3)
    axes[0].set_ylabel('$p/mc$', fontsize=label_size)
    # axes[0].set_xlabel('$\\xi$', fontsize=label_size)
    axes[0].set_xlabel('$r/r_g$', fontsize=label_size)
    axes[0].tick_params(axis="both", labelsize=tick_size)
    # axes[0].set_ylim([-5e4, 5e4])

    # efield, = ax2.plot(xs, E1*Delta_r(rxi(xs)), color='g', linewidth=0.6)
    efield, = ax2.plot(rxi(xs), E1*Delta_r(rxi(xs)), color='g', linewidth=0.6)

    axes[0].axvline(rl_inner, linestyle='--', color='r')
    axes[0].axvline(rl_outer, linestyle='--', color='r')
    axes[0].axvline(r_null, linestyle='--', color='magenta')

    axes[0].set_yscale('symlog', linthreshy=1000)
    # ax2.set_yscale('symlog')
    ax2.set_ylabel("$E$", fontsize=label_size)
    ax2.tick_params(axis="both", labelsize=tick_size)
    # ax2.set_ylim([-1e5, 3e4])
   
    ax22 = axes[1].twinx()

    # j, = ax22.plot(xs, J1*Delta_r(rxi(xs)) * 30)
    # rho, = ax22.plot(xs, (rho_e + rho_p))
    # ax22.plot(xi(rs), j0 * conf['B0'] * 30, "g--")
    # ax22.plot(xi(rs), rho0 * conf['B0'], "r--")
    j, = ax22.plot(rxi(xs), J1*Delta_r(rxi(xs)) * 30)
    rho, = ax22.plot(rxi(xs), (rho_e + rho_p))
    ax22.plot(rs, j0 * conf['B0'] * 30, "g--")
    ax22.plot(rs, rho0 * conf['B0'], "r--")

    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].tick_params(axis="x", labelsize=tick_size)
    ax22.tick_params(axis="y", labelsize=tick_size)
    # axes[1].set_xlabel('$\\xi$', fontsize=label_size)
    axes[1].set_xlabel('$r/r_g$', fontsize=label_size)
    txt = axes[1].text(0.7,1.06,"$t = {:.1f}$".format(num * conf['delta_t']),
                       size=30,transform = axes[1].transAxes)
    # ax22.set_ylim([-1e8, 1.5e8])

    fig.savefig("plots/%06d.png" % (num // 4000))
    plt.close(fig)


agents = 7
num_re = re.compile(r"\d+")
steps = [int(num_re.findall(f.stem)[0]) for f in Path(data_dir).glob("data*.h5")]
steps.sort()
print(steps)
for step in steps:
    plotfile = Path("plots/%06d.png" % (step // 4000))
    if plotfile.exists() and len(steps) > 0:
        steps.remove(step)
print(steps)

# make_plot(steps[10])
with Pool(processes=agents) as pool:
    pool.map(make_plot, steps)
