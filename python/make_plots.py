#!/usr/bin/python

import h5py
import matplotlib.pyplot as plt
import sys
import os

data_dir = sys.argv[1]
if not os.path.isdir(data_dir):
    print("Given path is not a directory")
    exit(0)

for n in range(250):
    # Change data directory!
    file_path = os.path.join(data_dir, "output%06d.h5" % (n * 20));
    f = h5py.File(file_path)
#     E1 = f["E1"]
#     J1 = f["J1"]
#     rho_e = f["Rho_e"]
#     rho_p = f["Rho_p"]
    x_e = f["Electrons_x"]
    p_e = f["Electrons_p"]
    x_p = f["Positrons_x"]
    p_p = f["Positrons_p"]
    plt.cla()
    plt.plot(x_e, p_e, '.')
    plt.plot(x_p, p_p, ".")
    plt.savefig("%06d.png" % n)
