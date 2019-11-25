#!/usr/bin/python
import h5py
import numpy as np
import pytoml
import sys
import os
from datalib_logsph import Data

class Grid:
    def __init__(self, conf):
        self.N1 = conf['Grid']['N'][0] / conf['Simulation']['downsample'] + 2*conf['Grid']['guard'][0]
        self.N2 = conf['Grid']['N'][1] / conf['Simulation']['downsample'] + 2*conf['Grid']['guard'][1]
        self.size1 = conf['Grid']['size'][0]
        self.size2 = conf['Grid']['size'][1]
        self.lower1 = conf['Grid']['lower'][0]
        self.lower2 = conf['Grid']['lower'][1]
        self.delta1 = self.size1 * conf['Simulation']['downsample'] / conf['Grid']['N'][0]
        self.delta2 = self.size2 * conf['Simulation']['downsample'] / conf['Grid']['N'][1]
        self.guard1 = conf['Grid']['guard'][0]
        self.guard2 = conf['Grid']['guard'][1]

    def find_field(self, point, B1, B2, B3):
        logr = np.log(point[0])
        c1 = int((logr - self.lower1) / self.delta1)
        c2 = int((point[1] - self.lower2) / self.delta2)
        if c1 < 0 or c1 >= 512 or c2 < 0 or c2 >= 512:
            return (0, 0, 0)
        x1 = logr - c1 * self.delta1
        x2 = point[1] - c2 * self.delta2
        vB1 = x1 * B1[c2, c1] + (1.0 - x1) * B1[c2, c1-1]
        vB2 = x2 * B2[c2, c1] + (1.0 - x2) * B2[c2-1, c1]
        vB3 = B3[c2, c1]
        return (vB1, vB2, vB3)

data_dir = sys.argv[1]
data = Data(data_dir)
conf = data._conf

if len(sys.argv) == 4:
    timestep = int(sys.argv[2])
    max_step = int(sys.argv[3])
    steps = np.arange(timestep, max_step)
else:
    timestep = 0
    steps = data.fld_steps

print(steps)

output_path = os.path.join(data_dir, 'field_lines/')
if not os.path.exists(output_path):
    os.mkdir(output_path)

num_lines = 6
max_points = 200

# rs = zeros(num_lines, max_points)
# thetas = zeros(num_lines, max_points)
# phis = zeros(num_lines, max_points)
grid = Grid(conf)
dt = 0.002
sim_dt = conf['delta_t']
data_interval = conf['Simulation']['data_interval']

phi_i = 0.0
for step in steps:
    data.load_fld(step)
    t = step * sim_dt * conf['Simulation']['data_interval']
    print("working on", step, "at", t)

    B1 = data.B1[()]
    B2 = data.B2[()]
    B3 = data.B3[()]

    if t <= 20.0:
        phi_i += conf['omega'] * (t / 25.0) * 0.5 * sim_dt * data_interval;
    elif t <= 40.0:
        phi_i += conf['omega'] * (1.0 - (t - 25.0) / 25.0) * 0.5 * sim_dt * data_interval;
    for n in range(num_lines):
        r_i = np.exp(conf['Grid']['lower'][0] + 0.02)
        theta_i = (n + 2.5) * np.pi * 0.35 / (num_lines + 5)
        # print('Line', n, 'starting at', (r_i, theta_i, phi_i))

        i = 0
        p = (r_i, theta_i, phi_i)
        ps = [(p[0]*np.sin(p[1])*np.cos(p[2]), p[0]*np.sin(p[1])*np.sin(p[2]),
               p[0]*np.cos(p[1]))]
        while p[0] > 1.02 and p[0] < 30:
            vB = grid.find_field(p, B1, B2, B3)
            vB /= np.sqrt(vB[0] * vB[0] + vB[1] * vB[1] + vB[2] * vB[2])
            p = (np.exp(np.log(p[0]) + vB[0]*dt), p[1] + vB[1]*dt, p[2] + vB[2]*dt)
            ps.append((p[0]*np.sin(p[1])*np.cos(p[2]), p[0]*np.sin(p[1])*np.sin(p[2]),
                       p[0]*np.cos(p[1])))
            # if n == 0: print(p)
        p_array = np.array(ps).flatten()
        # dphi = 0.5*(p_array[2] + p_array[-1])
        # p_array[2::3]-
        p_array.astype(np.float32).tofile(os.path.join(data_dir, 'field_lines/line_%d_%06d' % (n, step)))
    # print(f.keys())

