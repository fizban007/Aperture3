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
        meshfile = h5py.File(os.path.join(path, "mesh.h5"), "r", swmr=True)

        self.x1 = np.array(meshfile["x1"])
        self.x2 = np.array(meshfile["x2"])
        self.r = np.pad(
            np.exp(
                np.linspace(
                    0,
                    self.conf["Grid"]["size"][0],
                    conf["Grid"]["N"][0] // conf["Simulation"]["downsample"],
                )
                + self.conf["Grid"]["lower"][0]
            ),
            self.conf["Grid"]["guard"][0],
            "constant",
        )
        self.theta = np.pad(
            np.linspace(
                0,
                self.conf["Grid"]["size"][1],
                conf["Grid"]["N"][1] // conf["Simulation"]["downsample"],
            )
            + self.conf["Grid"]["lower"][1],
            self.conf["Grid"]["guard"][1],
            "constant",
        )

        meshfile.close()
        self.rv, self.thetav = np.meshgrid(self.r, self.theta)

        # Generate a list of output steps
        num_re = re.compile(r"\d+")
        self.steps = [
            int(num_re.findall(f.stem)[0]) for f in Path(path).glob("data*.h5")
        ]
        self.steps.sort()
        self.data_interval = self.steps[-1] - self.steps[-2]

    def load(self, step):
        if not step in self.steps:
            print("Step not in data directory!")
            return
        data_dir = self.path
        path = os.path.join(data_dir, f"data{step:06d}.h5")
        data = h5py.File(path, "r", swmr=True)

        self.rho_e = data["Rho_e"][()]
        self.rho_p = data["Rho_p"][()]
        self.rho_i = data["Rho_i"][()]
        self.pairs = data["pair_produced"][()]
        self.E1 = data["E1"][()]
        self.E2 = data["E2"][()]
        self.E3 = data["E3"][()]
        self.B1 = data["B1"][()] + data["B_bg1"][()]
        self.B2 = data["B2"][()] + data["B_bg2"][()]
        self.B3 = data["B3"][()] + data["B_bg3"][()]
        self.j1 = data["J1"][()]
        self.j2 = data["J2"][()]
        self.j3 = data["J3"][()]

        dtheta = (
            self.theta[self.conf["Grid"]["guard"][1] + 2]
            - self.theta[self.conf["Grid"]["guard"][1] + 1]
        )
        self.flux = np.cumsum(
            self.B1 * self.rv * self.rv * np.sin(self.thetav) * dtheta, axis=0
        )
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
