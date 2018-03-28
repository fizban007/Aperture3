# How to compile

First clone the directory:

    git clone https://github.com/fizban007/1Dpic
    
Now to build this, you need to have `cmake`, `boost`, and `hdf5` installed. To
compile this on `tigressdata`, use the following module load:

    module load rh/devtoolset/6
    module load boost/1.54.0
    module load openmpi/gcc/1.8.8
    module load hdf5/gcc/1.8.12
    
Then from the directory where you cloned the project, do:

    cd 1Dpic
    git checkout tigressdata
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

Typically I would run with something like this:

    mpirun -np 1 ./aperture -s 200000 -d 200
    
I recommend going through the `sim.conf` file to see what options are available,
and read `main.cpp` to see how to change initial conditions.

# How to plot

A simple python script is included in the `python` directory to make plots of
phase-space diagrams of particles. It accepts one parameter which is the data
directory. By default the simulation will output data to `Data` directory and
everything will be under a timestamped subdirectory. Say a simulation was done
on 2018-02-01 at 11:02am, then to plot its result, run:

    ./make_plots.py ../Data/Data20180201-1102
    
However, it is pretty rigid on how many frames to draw and at what interval. So
change those in the python script directly when you need to customize those.

To make a movie on `tigressdata` after you made the plots, do this in the python
directory:

    ffmpeg -y -f image2 -r 12 -i %06d.png -c:v libx264 -crf 18 -pix_fmt yuv420p movie.mp4
    
You can also changed the filename of the movie to whatever you like.
