import sys
import numpy
import logging
from astropy import units as u
from amuse.units import units
from galpy_amuse import compare_galpy_and_amuse
from galpy_amuse import convert_galpy_to_amuse_times

sys.path.insert(0, "/supaharris")
from data.parse_hilker_2019 import parse_hilker_2019_orbits
from data.parse_hilker_2019 import parse_hilker_2019_combined


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(message)s")
    logger = logging.getLogger(__file__)
    logger.info("Running {0}".format(__file__))

    # Hilker+ 2019 --> Orbital parameters for 154 MW GCs
    h19_orbits = parse_hilker_2019_orbits(logger)
    h19_combined = parse_hilker_2019_combined(logger)

    # Data needed for the initial conditions
    gc_name = "NGC 104"
    igc, = numpy.where(h19_orbits["Cluster"] == gc_name)[0]
    imatch, = numpy.where(h19_combined["Cluster"] == gc_name)[0]

    # Simulation runtime
    ts = numpy.linspace(0.0, 1, 8096+1) * u.Gyr
    tstart, tend, Nsteps, dt = convert_galpy_to_amuse_times(ts)

    # Here we define the function that operates on the particle data during the simulation
    com = numpy.zeros((Nsteps,3))
    times = numpy.zeros(Nsteps)
    def calculate_com(stars, time, i):
        # print_progressbar(i, Nsteps)
        c = stars.center_of_mass().value_in(units.parsec)
        com[i][:] = c
        times[i] = time.value_in(units.Myr)
        return stars

    # Do the orbit integrations with Galpy (analytical, fast)
    # as well as with AMUSE/Gadget2 (Nbody, expensive)
    compare_galpy_and_amuse(
        logger,
        h19_orbits[igc],
        h19_combined[imatch],
        N=1, ts=ts, do_something=calculate_com,
        number_of_workers=1,
    )
