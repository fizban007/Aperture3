# How to compile

First clone the directory:

    git clone https://github.com/fizban007/1Dpic
    
Now to build this, you need to have `cmake`, `boost`, and `hdf5` installed. To
compile this on `tigressdata`, use the following module load:

    module load rh/devtoolset/6
    module load boost/1.54.0
    module load openmpi
    module load hdf5/gcc/1.8.12
    
Then from the directory where you cloned the project, do:

    cd 1Dpic
    mkdir build
    cd build
    cmake -DBoost_NO_BOOST_CMAKE=true -DBoost_NO_SYSTEM_PATHS=true -DBOOST_ROOT:PATHNAME=/usr/local/boost/1.54.0 ..
    make
    
Now there will be a new executable file `aperture` under `1Dpic/bin/`. You can
run it to run the code. The executable accepts the following arguments:

    -h, --help          Prints this help message. (default: false)
    -c, --config arg    Configuration file for the simulation. (default:
                        sim.conf)
    -s, --steps arg     Number of steps to run the simulation. (default: 2000)
    -d, --interval arg  The interval to output data to the hard disk. (default:
                        20)
    -x, --dimx arg      The number of processes in x direction. (default: 1)
    
I recommend going through the `sim.conf` file to see what options are available,
and read `main.cpp` to see how to change initial conditions.
