#!/usr/bin/env python

import h5py
import numpy as np
import pytoml
from pathlib import Path
import os
import glob
import re

class Data:
    def __init__(self, path):
        conf = self.load_conf(path)
        N1 = (
            conf["Grid"]["N"][0] // conf["Simulation"]["downsample"]
            + conf["Grid"]["guard"][0] * 2
        )
        N2 = (
            conf["Grid"]["N"][1] // conf["Simulation"]["downsample"]
            + conf["Grid"]["guard"][1] * 2
        )
        self.conf = conf
        self.path = path
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

        # Load mesh file
        meshfile = h5py.File(os.path.join(self.data_dir, "mesh.h5"), "r", swmr=True)

        self.x1 = mesh["x1"].value
        self.x2 = mesh["x2"].value

        meshfile.close()

        # Generate a list of output steps
        num_re = re.compile(r'\d+')
        self.steps = [ int(regex.findall(f.stem)[0]) for f in Path(path).glob('data*.h5') ]
        self.steps.sort()
        self.data_interval = self.step[-1] - self.step[-2]

    def load(self, fname):
        data_dir = self.path
        path = os.path.join(data_dir, fname)
        data = h5py.File(path, "r", swmr=True)

        self.rho_e = data["Rho_e"].value
        self.rho_p = data["Rho_p"].value
        self.rho_i = data["Rho_i"].value
        self.pairs = data["pair_produced"].value
        self.E1 = data["E1"].value
        self.E2 = data["E2"].value
        self.E3 = data["E3"].value
        self.B1 = data["B1"].value + data["B_bg1"].value
        self.B2 = data["B2"].value + data["B_bg2"].value
        self.B3 = data["B3"].value + data["B_bg3"].value
        self.j1 = data["J1"].value
        self.j2 = data["J2"].value
        self.j3 = data["J3"].value
        self.flux = data["flux"].value
        self.B = np.sqrt(self.B1 * self.B1 + self.B2 * self.B2 + self.B3 * self.B3)
        self.EdotB = (
            self.E1 * self.B1 + self.E2 * self.B2 + self.E3 * self.B3
        ) / self.B

        data.close()

    def load_conf(self, path):
        conf_path = os.path.join(path, "config.toml")
        with open(conf_path, "rb") as f:
            conf = pytoml.load(f)
            return conf
