APERTURE &nbsp;&nbsp;![](https://github.com/fizban007/Aperture3/blob/master/logo.png "Aperture Logo")
=============

# How to compile

First clone the directory:

    git clone https://github.com/fizban007/Aperture3
    
Now to build this, you need to have `cmake`, `cuda`, and `hdf5` installed. To
compile this on `tigergpu`, use the following module load:

    module load rh/devtoolset/6
    module load cudatoolkit
    module load openmpi/cuda-9.0/gcc
    module load hdf5/gcc
    
A configuration of the code is called a "lab". Once you clone the code, you can
see that there is a directory called `labs`. Each lab is a setup to run a
specific simulation, provides their own `main.cpp`, and can override any number
of source files or supply new source files.

To compile the code, either go to a lab directory and run:

    ./buick cmake
    ./buick
    
`buick` is a wrapper around `cmake` and `make`. It streamlines the source-file
overriding of the specific simulation setup, and keeps different setups
relatively independent.

Configurations of the simulation can be found in `config.toml` under each lab.
Read through the file to see what can be changed without recompiling.
