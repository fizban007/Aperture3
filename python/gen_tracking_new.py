#!/usr/bin/python
import h5py
import numpy as np
import pytoml
import sys
import os
from datalib_logsph import Data

data_dir = sys.argv[1]
if not os.path.isdir(data_dir):
    print("Given path is not a directory")
    exit(0)

data = Data(data_dir)

if len(sys.argv) == 4:
    timestep = int(sys.argv[2])
    max_step = int(sys.argv[3])
    steps = np.arange(timestep, max_step)
else:
    timestep = 0
    # print(conf.keys())
    steps = data.ptc_steps

print(steps)

output_path = os.path.join(data_dir, 'tracking/')
if not os.path.exists(output_path):
    os.mkdir(output_path)

for step in steps:
    print("working on", step)
    data.load_ptc(step)
    # f = h5py.File(fname)
    e_x1 = np.exp(data.electron_x1[()])
    e_x2 = data.electron_x2[()]
    e_x3 = data.electron_x3[()]
    p_x1 = np.exp(data.positron_x1[()])
    p_x2 = data.positron_x2[()]
    p_x3 = data.positron_x3[()]
    i_x1 = np.exp(data.ion_x1[()])
    i_x2 = data.ion_x2[()]
    i_x3 = data.ion_x3[()]

    pos_e = np.zeros(3*len(e_x1))
    pos_p = np.zeros(3*len(p_x1) + 3*len(i_x1))
    pos_e[::3] = e_x1 * np.sin(e_x2) * np.cos(e_x3)
    pos_e[1::3] = e_x1 * np.sin(e_x2) * np.sin(e_x3)
    pos_e[2::3] = e_x1 * np.cos(e_x2)
    pos_p[:3*len(p_x1):3] = p_x1 * np.sin(p_x2) * np.cos(p_x3)
    pos_p[1:3*len(p_x1):3] = p_x1 * np.sin(p_x2) * np.sin(p_x3)
    pos_p[2:3*len(p_x1):3] = p_x1 * np.cos(p_x2)
    pos_p[3*len(p_x1)::3] = i_x1 * np.sin(i_x2) * np.cos(i_x3)
    pos_p[3*len(p_x1)+1::3] = i_x1 * np.sin(i_x2) * np.sin(i_x3)
    pos_p[3*len(p_x1)+2::3] = i_x1 * np.cos(i_x2)

    pos_e.astype(np.float32).tofile(os.path.join(data_dir, 'tracking/pos_e_%06d' % step))
    pos_p.astype(np.float32).tofile(os.path.join(data_dir, 'tracking/pos_p_%06d' % step))
    # print(f.keys())

