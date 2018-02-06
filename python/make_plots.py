#!/usr/bin/python

import h5py
import matplotlib.pyplot as plt

for n in range(200):
    # Change data directory!
    f = h5py.File("../Data/Data20180206-1214/output%06d.h5" % (n * 10))
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
    plt.savefig("%06d.png" % (n * 10))
