## Install & use instructions

To run this Docker environment (and all the code in it), you only need to install
the latest versions of `Docker` and `docker-compose`. 

To start the Docker container (incl. serving Jupyter notebook at  `localhost:8888`),
run the following command in the top level of this repository:

`docker-compose up --build`

And to start the container shell:

`docker exec -it notebook bash`

See [the Docker documentation](https://docs.docker.com/) and [commonly used commands](https://towardsdatascience.com/15-docker-commands-you-should-know-970ea5203421).


# Freya (20200326)

Below sits the setup currently used to run the simulations on Freya.
To run a script interactively on login node for testing purposes: `setup_tidalshocks; set_interactive`.
To run a script via the job scheduler (SLURM): first use `setup_tidalshocks` in the script

### Currently Loaded Modulefiles:
```text
  1) git/2.16             8) mpi4py/3.0.0
  2) anaconda/3/2019.03   9) gsl/2.4
  3) cmake/3.13          10) fftw-mpi/3.3.8
  4) ffmpeg/3.4          11) hdf5-mpi/1.8.21
  5) intel/19.0.5        12) netcdf-mpi/4.4.1
  6) mkl/2019.5          13) jdk/8
  7) impi/2019.5         14) cuda/10.1
```


### `~/.bashrc`

```bash

set -o noclobber
set -o vi 

module load git
module load anaconda/3/2019.03

# For AMUSE
module load cmake

# For mpirun interactively on login nodes
# set_interactive

# For videos
module load ffmpeg

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/lib64:/lib:/usr/lib:/usr/lib64

# Add LaTeX
export PATH=/u/timoh/texlive/2018/bin/x86_64-linux:${PATH}

# Add Julia
export PATH=/u/timoh/julia-1.1.1/bin/:${PATH}
```


### `~/.bash_functions`

```bash
set_interactive() {
    unset I_MPI_HYDRA_BOOTSTRAP
    unset I_MPI_PMI_LIBRARY
}

setup_tidalshocks() {
    export PYTHONCONFIG=python3-config

    module load intel
    module load mkl
    module load impi
    module load mpi4py
    export FC=ifort 
    export F90=ifort 
    export F77=ifort 
    export CC=icc 
    export CXX=icpc 
    export MPICC=mpiicc
    export MPICXX=mpiicpc
    export MPIFC=mpiifort
    export MPIF90=mpiifort
    export MPIF77=mpiifort

    module load gsl
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GSL_HOME}/lib
    export CFLAGS="${CFLAGS} $(gsl-config --cflags)"
    export LDFLAGS="${LDFLAGS} $(gsl-config --libs)"

    module load cmake
    module load fftw-mpi
    module load hdf5-mpi
    module load netcdf-mpi
    module load jdk
    module load cuda
    export CUDA_TK=$CUDA_HOME
    export NVCC=$CUDA_HOME/bin/nvcc

    cd /u/timoh/phd/tidalshocks
    source activate tidalshocks
    # export AMUSE_DIR=/u/timoh/phd/amuse/
    # export PYTHONPATH=${PYTHONPATH}:${AMUSE_DIR}test:${AMUSE_DIR}src
     
    export IPYTHONDIR=/u/timoh/conda-envs/tidalshocks/.ipython
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/timoh/conda-envs/tidalshocks/lib
    export PKG_CONFIG_PATH=/u/timoh/conda-envs/tidalshocks/lib/pkgconfig/

    export AMUSE_DIR=/u/timoh/phd/amuse
    export PYTHONDEV_CFLAGS="$( $PYTHONCONFIG --cflags ) -I$(python -c 'import numpy; print(numpy.get_include())')"
    export PYTHONDEV_LDFLAGS=$( $PYTHONCONFIG --ldflags )
    export CYTHON=/u/timoh/conda-envs/tidalshocks/bin/cython

    module list
    source $INTEL_HOME/bin/iccvars.sh intel64
    source $I_MPI_ROOT/intel64/bin/mpivars.sh
}
```


### AMUSE installation
- `./configure --with-hdf5=$HDF5_HOME/bin/h5pcc --with-fftw=$FFTW_HOME 
    --with-gsl-prefix=$GSL_HOME --with-cuda-libdir=$CUDA_HOME/lib 
    --enable-cuda --enable-mpi MPIEXEC=mpiexec.hydra`
- `make DOWNLOAD_CODES=1`
- `python setup.py install`

### Galpy installation
- Modified `setup.py` to pick up the location of gsl b/c for some reason it 
    ignores all env vars :/. See the code block below.
- `python setup.py install --compiler=intel64`

```python
cmd= ['gsl-config',
      '--cflags']
gsl_cflags= subprocess.Popen(cmd,shell=sys.platform.startswith('win'),
                             stdout=subprocess.PIPE).communicate()[0].strip()
if PY3:
    gsl_cflags= gsl_cflags.decode('utf-8')
extra_compile_args.append(gsl_cflags)

cmd= ['gsl-config',
      '--libs']
gsl_libs= subprocess.Popen(cmd,shell=sys.platform.startswith('win'),
                           stdout=subprocess.PIPE).communicate()[0].strip()
if PY3:
    gsl_libs= gsl_libs.decode('utf-8')
gsl_libs= gsl_libs.replace("-L/", "/").replace(" -lgsl -lgslcblas -lm", "")
library_dirs.append(gsl_libs)
```
