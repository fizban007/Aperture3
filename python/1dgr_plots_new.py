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
import pytoml
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

coeffile = h5py.File(Path(data_dir) / "../coef.h5")

rl_inner = coeffile["rL_inner"][()]
rl_outer = coeffile["rL_outer"][()]
rs = coeffile["r"][()]
rho0 = coeffile["rho"][()]
j0 = coeffile["ju1"][()]
th = coeffile["th"][()]
r_null = rs[np.argmin(abs(rho0))]

coeffile.close()

conf = data.conf

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


label_size = 30
tick_size = 20

js = abs(np.interp(rxi(xs), rs, j0 * conf["B0"]))
thetas = np.interp(rxi(xs), rs, th)

def make_plot(num):
    print("Working on", num)
    data.load(num)
    E1 = data.E1
    J1 = data.J1
    rho_e = data.Rho_e
    rho_p = data.Rho_p
    x_e = data.electron_x
    p_e = np.sign(data.electron_p) * abs(data.electron_E)
    x_p = data.positron_x
    p_p = np.sign(data.positron_p) * abs(data.positron_E)
    x_ph = data.ph_x1
    # p_ph = data.ph_p1
    p_ph = np.sign(data.ph_p1) * abs(data.ph_E)

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(18, 17)
    ax2 = axes[0, 0].twinx()

    el, = axes[0, 0].plot(
        # rxi(x_e), p_e / Delta_r(rxi(x_e)), ".", markersize=1.0, color="blue", alpha=0.3
        rxi(x_e), p_e, ".", markersize=1.0, color="blue", alpha=0.3
    )
    po, = axes[0, 0].plot(
        rxi(x_p),
        # p_p / Delta_r(rxi(x_p)),
        p_p,
        ".",
        markersize=1.0,
        color="orange",
        alpha=0.3,
    )
    ph, = axes[0, 0].plot(
        # rxi(x_ph), p_ph / Delta_r(rxi(x_ph)), ".", markersize=1.0, color="k", alpha=0.3
        rxi(x_ph), p_ph, ".", markersize=1.0, color="k", alpha=0.3
    )
    axes[0, 0].set_ylabel("$p/mc$", fontsize=label_size)
    axes[0, 0].set_xlabel("$r/r_g$", fontsize=label_size)
    axes[0, 0].tick_params(axis="both", labelsize=tick_size)
    # axes[0].set_ylim([-5e4, 5e4])

    efield, = ax2.plot(rxi(xs), E1 * Delta_r(rxi(xs)), color="g", linewidth=0.6)

    axes[0, 0].axvline(rl_inner, linestyle="--", color="r")
    axes[0, 0].axvline(rl_outer, linestyle="--", color="r")
    axes[0, 0].axvline(r_null, linestyle="--", color="magenta")

    # axes[0, 0].set_yscale("symlog", linthreshy=1000)
    axes[0, 0].set_yscale("symlog")
    # ax2.set_yscale('symlog')
    ax2.set_ylabel("$E$", fontsize=label_size)
    ax2.tick_params(axis="both", labelsize=tick_size)
    ax2.yaxis.get_offset_text().set_fontsize(tick_size)
    # ax2.set_ylim([-1e5, 3e4])

    ax22 = axes[0, 1].twinx()

    j, = ax22.plot(rxi(xs), J1 * Delta_r(rxi(xs)) * 30)
    rho, = ax22.plot(rxi(xs), (rho_e + rho_p))
    ax22.plot(rs, j0 * conf["B0"] * 30, "k--")
    ax22.plot(rs, rho0 * conf["B0"], "g--")

    axes[0, 1].tick_params(axis="y", labelleft=False)
    axes[0, 1].tick_params(axis="x", labelsize=tick_size)
    ax22.tick_params(axis="y", labelsize=tick_size)
    ax22.yaxis.get_offset_text().set_fontsize(tick_size)
    # axes[1].set_xlabel('$\\xi$', fontsize=label_size)
    axes[0, 1].set_xlabel("$r/r_g$", fontsize=label_size)
    txt = axes[0, 1].text(
        0.7,
        1.06,
        "$t = {:.2f}$".format(
            num * conf["Simulation"]["data_interval"] * conf["delta_t"]
        ),
        size=30,
        transform=axes[0, 1].transAxes,
    )
    # ax22.set_ylim([-1e8, 1.5e8])
    #
    a, b, hist_e = axes[1, 0].hist(
        np.log10(np.minimum(abs(p_e[np.nonzero(p_e)]), 1e10) + 1e-4), bins=100, histtype=u"step"
    )
    a, b, hist_p = axes[1, 0].hist(
        np.log10(np.minimum(abs(p_p[np.nonzero(p_p)]), 1e10) + 1e-4), bins=100, histtype=u"step"
    )
    a, b, hist_ph = axes[1, 0].hist(
        np.log10(np.minimum(abs(p_ph[np.nonzero(p_ph)]), 1e10) + 1e-4), bins=100, histtype=u"step"
    )
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_ylabel("$u\,dN/du$", fontsize=label_size)
    axes[1, 0].set_xlabel("$log_{10}u$", fontsize=label_size)
    axes[1, 0].tick_params(axis="both", labelsize=tick_size)
    axes[1, 0].set_xlim([0, np.log10(2000/conf["e_min"])])

    # axes[1, 1].plot(rxi(xs), (rho_p - rho_e)/js)
    axes[1, 1].plot(rxi(xs),(rho_p - rho_e) * alpha(rxi(xs), thetas) / js / np.sqrt(grr(rxi(xs), thetas)))
    axes[1, 1].plot(rxi(xs),np.ones(len(xs)))
    axes[1, 1].set_ylabel("M", fontsize=label_size)
    axes[1, 1].set_xlabel("$r/r_g$", fontsize=label_size)
    axes[1, 1].tick_params(axis="both", labelsize=tick_size)
    axes[1, 1].set_ylim([0, 20])

    fig.savefig("plots/%05d.png" % num)
    plt.close(fig)


agents = 7
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
