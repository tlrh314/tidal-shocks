def print_info():
    import subprocess, numpy, scipy, mpi4py

    python_version = subprocess.Popen(["python", "--version"], shell=False,
        stdout=subprocess.PIPE).communicate()[0].decode().strip()
    openmpi_version = subprocess.Popen(["mpicxx", "-v"], shell=False,
        stdout=subprocess.PIPE).communicate()[0].decode().strip().split("\n")[0]
    mpirun_version = subprocess.Popen(["mpirun", "--version"], shell=False,
        stdout=subprocess.PIPE).communicate()[0].decode().strip().split("\n")[0]
    mpiexec_version = subprocess.Popen(["mpiexec", "--version"], shell=False,
        stdout=subprocess.PIPE).communicate()[0].decode().strip().split("\n")[0]
    pip_freeze = subprocess.Popen(["pip", "freeze"], shell=False,
        stdout=subprocess.PIPE).communicate()[0].decode().strip()

    print(python_version)
    print(openmpi_version)
    print(mpirun_version)
    print(mpiexec_version)
    for p in pip_freeze.split("\n"):
        if "amuse" in p: print(p)
    print("NumPy:", numpy.__version__)
    print("SciPy:", scipy.__version__)
    print("mpi4py:", mpi4py.__version__)


def test_bhtree(number_of_workers):
    from amuse.units import units, nbody_system
    from amuse.community.bhtree.interface import BHTreeInterface, BHTree
    convert_nbody = nbody_system.nbody_to_si(5.0 | units.kg, 10.0 | units.m)
    instance = BHTree(convert_nbody, number_of_workers=number_of_workers)
    instance.commit_parameters()

    indices = instance.new_particle(
        [15.0, 30.0] | units.kg,
        [10.0, 20.0] | units.m, [20.0, 40.0] | units.m, [30.0, 50.0] | units.m,
        #1.0 | units.m/units.s, 1.0 | units.m/units.s, 3.0 | units.m/units.s
        [0.0, 0.01] | units.m/units.s, [0.0, 0.01] | units.m/units.s, [0.0, 0.01] | units.m/units.s,
        [10.0, 20.0] | units.m
    )
    instance.commit_particles()

    instance.get_mass([4,5])
    # AmuseException --> "Error when calling 'get_mass' of a 'BHTree', errorcode is -1"


def test_brutus(number_of_workers):
    from amuse.community.brutus.interface import Brutus
    gravity = Brutus(number_of_workers=number_of_workers, redirection="none")
    gravity.set_brutus_output_directory("folder/does/not/exist")
    gravity.commit_parameters()
    print(gravity.parameters)


def test_mercury(number_of_workers):
    from amuse.units import units
    from amuse.ext.solarsystem import new_solar_system
    from amuse.community.mercury.interface import Mercury

    solar_system = new_solar_system()

    mercury = Mercury(number_of_workers=number_of_workers, redirection="none")
    mercury.particles.add_particles(solar_system)
    mercury.parameters.info_file = "folder/does/not/exists"
    mercury.parameters.timestep = (1.|units.day)
    print(mercury.parameters)

    mercury.evolve_model(1.|units.yr)
    mercury.stop()


def test_gadget(number_of_workers):
    from amuse.units import units, nbody_system
    from amuse.ic.plummer import new_plummer_sphere

    from amuse.community.gadget2.interface import Gadget2
    gravity = Gadget2(
        number_of_workers=number_of_workers,
        channel_type="sockets",
        redirection="none"
    )
    gravity.parameters.gadget_output_directory = "folder/does/not/exist"
    gravity.commit_parameters()


if __name__ == "__main__":
    print_info()

    from amuse.support.exceptions import CodeException
    from amuse.support.exceptions import AmuseException

    try:
        test_gadget(1)  # no bug
    except CodeException as e:
        print("Code throws error")
    try:
        test_gadget(4)  # yes bug
    except CodeException as e:
        print("Worker dies without communicating this back to AMUSE")
        raise  # never reached



    # try:
    #     test_mercury(1)  # no bug, but code complains file does not exists and returns (does not die)
    # except Exception as e:
    #     print("Code throws error")
    # test_mercury(4)  # no bug

    # try:
    #     test_bhtree(1)  # no bug
    # except AmuseException as e:
    #     print("Code throws error")
    # try:
    #     test_bhtree(4)  # no bug
    # except AmuseException as e:
    #     print("Code throws error")

    # try:
    #     test_brutus(1)  # no bug
    # except CodeException as e:
    #     print("Code throws error")
    # try:
    #     test_brutus(4)  # no bug
    # except CodeException as e:
    #     print("Code throws error")
    #     raise
