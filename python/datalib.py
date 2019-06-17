#!/usr/bin/env python

import h5py
import numpy as np
import pytoml
from pathlib import Path
import os
import glob
import re


class Data:
    _coord_keys = ["x1", "x2", "r", "theta", "rv", "thetav", "dr", "dtheta"]

    def __init__(self, path):
        conf = self.load_conf(path)
        self._conf = conf
        self._path = path

        # Load mesh file
        self._meshfile = os.path.join(path, "mesh.h5")

        # Generate a list of output steps
        num_re = re.compile(r"\d+")
        self.steps = [
            int(num_re.findall(f.stem)[0]) for f in Path(path).glob("data*.h5")
        ]
        self.steps.sort()
        self.data_interval = self.steps[-1] - self.steps[-2]
        self._current_step = self.steps[0]
        self._mesh_loaded = False
        f = h5py.File(
            os.path.join(self._path, f"data{self._current_step:06d}.h5"), "r", swmr=True
        )
        self._keys = list(f.keys()) + ["flux", "B"] + Data._coord_keys
        self.__dict__.update(("_" + k, None) for k in self._keys)
        f.close()

    def __getattr__(self, key):
        if key in self._keys:
            content = getattr(self, "_" + key)
            if content is not None:
                return content
            else:
                self._load_content(key)
                return getattr(self, "_" + key)
        elif key == "keys":
            return self._keys
        else:
            return None

    def load(self, step):
        if not step in self.steps:
            print("Step not in data directory!")
            return
        self._current_step = step
        for k in self._keys:
            if k not in Data._coord_keys:
                setattr(self, "_" + k, None)
        # self._mesh_loaded = False

    def _load_content(self, key):
        path = os.path.join(self._path, f"data{self._current_step:06d}.h5")
        data = h5py.File(path, "r", swmr=True)
        if key == "flux":
            self._load_mesh()
            dtheta = (
                self._theta[self._conf["Grid"]["guard"][1] + 2]
                - self._theta[self._conf["Grid"]["guard"][1] + 1]
            )
            self._flux = np.cumsum(
                self.B1 * self._rv * self._rv * np.sin(self._thetav) * dtheta, axis=0
            )
        elif key == "B":
            self._B = np.sqrt(self.B1 * self.B1 + self.B2 * self.B2 + self.B3 * self.B3)
        elif key in Data._coord_keys:
            self._load_mesh()
        elif key == "B1":
            setattr(self, "_" + key, data["B1"][()] + data["B_bg1"][()])
        elif key == "B2":
            setattr(self, "_" + key, data["B2"][()] + data["B_bg2"][()])
        elif key == "B3":
            setattr(self, "_" + key, data["B3"][()] + data["B_bg3"][()])
        # elif key == "EdotB":
        #     setattr(self, "_" + key, data["EdotBavg"][()])
        else:
            setattr(self, "_" + key, data[key][()])
        data.close()

    def _load_mesh(self):
        if self._mesh_loaded:
            return
        meshfile = h5py.File(self._meshfile, "r", swmr=True)

        self._x1 = meshfile["x1"][()]
        self._x2 = meshfile["x2"][()]
        self._r = np.pad(
            np.exp(
                np.linspace(
                    0,
                    self._conf["Grid"]["size"][0],
                    self._conf["Grid"]["N"][0]
                    // self._conf["Simulation"]["downsample"],
                )
                + self._conf["Grid"]["lower"][0]
            ),
            self._conf["Grid"]["guard"][0],
            "constant",
        )
        self._theta = np.pad(
            np.linspace(
                0,
                self._conf["Grid"]["size"][1],
                self._conf["Grid"]["N"][1] // self._conf["Simulation"]["downsample"],
            )
            + self._conf["Grid"]["lower"][1],
            self._conf["Grid"]["guard"][1],
            "constant",
        )

        meshfile.close()
        self._rv, self._thetav = np.meshgrid(self._r, self._theta)
        self._dr = (
            self._r[self._conf["Grid"]["guard"][0] + 2]
            - self._r[self._conf["Grid"]["guard"][0] + 1]
        )
        self._dtheta = (
            self._theta[self._conf["Grid"]["guard"][1] + 2]
            - self._theta[self._conf["Grid"]["guard"][1] + 1]
        )

        self._mesh_loaded = True

    def load_conf(self, path):
        conf_path = os.path.join(path, "config.toml")
        with open(conf_path, "rb") as f:
            conf = pytoml.load(f)
            return conf
