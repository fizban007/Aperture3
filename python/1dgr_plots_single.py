#!/usr/bin/env python

import matplotlib

matplotlib.use("Agg")
matplotlib.rc("text", usetex=True)
matplotlib.rc("font", family="serif")

import sys
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool
from datalib_1dgr import Data
import numpy as np
import h5py
import glob
import re


rp = 1.14107
rm = 0.858933
a = 0.99
theta = 0.44
cos_th = np.cos(theta)
sin_th = np.sin(theta)

data_dir = sys.argv[1]

data = Data(data_dir)
xs = data.x

coeffile = h5py.File("coef.h5")

rl_inner = coeffile["rL_inner"][()]
rl_outer = coeffile["rL_outer"][()]
rs = coeffile["r"][()]
rho0 = coeffile["rho"][()]
j0 = coeffile["ju1"][()]
th = coeffile["th"][()]
r_null = rs[np.argmin(abs(rho0))]

coeffile.close()

conf = data.conf
B0 = conf["B0"]

def xi(r):
    return np.log((r - rp) / (r - rm)) / (rp - rm)

def rxi(xi):
    exp_xi = np.exp(xi * (rp - rm))
    return (rp - rm * exp_xi) / (1.0 - exp_xi)

def Delta_r(r):
    return r * r - 2.0 * r + a * a

def Sigma_r(r):
    return r * r + a * a * cos_th * cos_th

def A_r(r):
    return np.sqrt(r * r + a * a) - Delta_r(r) * a * a * sin_th * sin_th

def Sigma(r, theta):
    cos_th = np.cos(theta)
    return r * r + a * a * cos_th * cos_th

def A(r, theta):
    sin_th = np.sin(theta)
    return (r * r + a * a)**2 - Delta_r(r) * a * a * sin_th * sin_th

def alpha(r, theta):
    return np.sqrt(Delta_r(r) * Sigma(r, theta) / A(r, theta))

def grr(r, theta):
    return Sigma(r, theta) / Delta_r(r)


label_size = 42
tick_size = 30

js = abs(np.interp(rxi(xs), rs, j0 * conf["B0"]))
thetas = np.interp(rxi(xs), rs, th)

def make_plot(num):
    print("Working on", num)
    data.load(num)

    E1 = data.E1
    x_e = data.electron_x
    p_e = np.sign(data.electron_p) * abs(data.electron_E)
    x_p = data.positron_x
    p_p = np.sign(data.positron_p) * abs(data.positron_E)
    x_ph = data.ph_x1
    # p_ph = data.ph_p1
    p_ph = np.sign(data.ph_p1) * abs(data.ph_E)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 12)
    ax2 = ax.twinx()

    el, = ax.plot(
        # rxi(x_e), p_e / Delta_r(rxi(x_e)), ".", markersize=1.0, color="blue", alpha=0.3
        rxi(x_e), p_e, ".", markersize=1.0, color="blue", alpha=0.3
    )
    po, = ax.plot(
        rxi(x_p),
        # p_p / Delta_r(rxi(x_p)),
        p_p,
        ".",
        markersize=1.0,
        color="orange",
        alpha=0.3,
    )
    ph, = ax.plot(
        # rxi(x_ph), p_ph / Delta_r(rxi(x_ph)), ".", markersize=1.0, color="k", alpha=0.3
        rxi(x_ph), p_ph, ".", markersize=1.0, color="k", alpha=0.3
    )
    ax.set_ylabel("$p/mc$", fontsize=label_size)
    ax.set_xlabel("$r/r_g$", fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.set_ylim(-2e5, 2e5)

    # axes[0].set_ylim([-5e4, 5e4])

    efield, = ax2.plot(rxi(xs), E1 * Delta_r(rxi(xs)) / B0, color="g", linewidth=1.2)

    ax.axvline(rl_inner, linestyle="--", color="r")
    ax.axvline(rl_outer, linestyle="--", color="r")
    ax.axvline(r_null, linestyle="--", color="magenta")

    # axes[0, 0].set_yscale("symlog", linthreshy=1000)
    ax.set_yscale("symlog")
    # ax2.set_yscale('symlog')
    ax2.set_ylabel("$E/B_0$", fontsize=label_size)
    ax2.tick_params(axis="both", labelsize=tick_size)
    ax2.yaxis.get_offset_text().set_fontsize(tick_size)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(-1,1))
    ax2.set_ylim(-2e-3, 2e-3)

    time = num * data.conf["delta_t"] * data.conf["Simulation"]["data_interval"]
    ax.text(7, 1e6, f"Time $= {time:.2f}r_g/c$", size=label_size)
    fig.savefig("plots/%05d.png" % num)
    plt.close(fig)


agents = 5
num_re = re.compile(r"\d+")
orig_steps = data.fld_steps
print(orig_steps)
steps = orig_steps.copy()
for step in orig_steps:
    plotfile = Path("plots/%05d.png" % step)
    if plotfile.exists() and len(steps) > 0:
        steps.remove(step)
    if len(sys.argv) > 2 and step > int(sys.argv[2]):
        steps.remove(step)
print(steps)

with Pool(processes=agents) as pool:
    pool.map(make_plot, steps)
