def print_info():
    import subprocess, numpy, scipy, mpi4py

    pyversion = subprocess.Popen(["python", "--version"], shell=False,
        stdout=subprocess.PIPE).communicate()[0].decode().strip()
    openmpiversion = subprocess.Popen(["mpicxx", "-v"], shell=False,
        stdout=subprocess.PIPE).communicate()[0].decode().strip().split("\n")[0]
    mpirunversion = subprocess.Popen(["mpirun", "--version"], shell=False,
        stdout=subprocess.PIPE).communicate()[0].decode().strip().split("\n")[0]
    mpiexecversion = subprocess.Popen(["mpiexec", "--version"], shell=False,
        stdout=subprocess.PIPE).communicate()[0].decode().strip().split("\n")[0]

    print(pyversion)
    print(openmpiversion)
    print(mpirunversion)
    print(mpiexecversion)
    print("NumPy:", numpy.__version__)
    print("NumPy:", numpy.__version__)
    print("mpi4py:", mpi4py.__version__)


def minimum_example(number_of_workers):
    converter = nbody_system.nbody_to_si(1 | units.MSun, 1 | units.parsec)

    particles = new_plummer_sphere(10, convert_nbody=converter)

    gravity = Gadget2(
        unit_converter=converter,
        number_of_workers=number_of_workers,
        redirection="none"
    )
    gravity.parameters.gadget_output_directory = "folder/does/not/exist"
    print(gravity.parameters)
    gravity.particles.add_particles(particles)

    gravity.evolve_model(1 | units.Myr)


if __name__ == "__main__":
    print_info()

    from amuse.units import units, nbody_system
    from amuse.ic.plummer import new_plummer_sphere
    from amuse.support.exceptions import CodeException
    from amuse.community.gadget2.interface import Gadget2
    try:
        minimum_example(1)  # no bug
    except CodeException as e:
        print("Code throws error")

    try:
        minimum_example(4)  # yes bug
    except CodeException as e:
        print("Code does silently and gets stuck in infinite loop...")
