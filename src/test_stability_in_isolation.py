import os
import re
import sys
import glob
import numpy
import argparse
import platform
from matplotlib import pyplot
pyplot.style.use("tlrh")
if "freya" in platform.node(): pyplot.switch_backend("agg")
from amuse.units import units
from amuse.io import write_set_to_file
from amuse.io import read_set_from_file


BASEDIR = "/u/timoh/phd" if "freya" in platform.node() else ""


def analyse_isolation(obs, model_name, rmin=1e-3, rmax=1e3, Nbins=256, smooth=False):
    from galpy_amuse_wrapper import print_stuff
    from galpy_amuse_wrapper import plot_SigmaR_vs_R

    # REMOVE
    limepy_model, limepy_sampled, amuse_sampled, converter = \
        obs.sample_deBoer2019_bestfit_king(Nstars=1000)
    # END REMOVE

    snap_base = "{}{}_{}_isolation_*.h5".format(obs.outdir, obs.gc_slug, model_name)
    snapshots = sorted(glob.glob(snap_base), key=lambda s:
        [ int(c) for c in re.split('(\d+)', s) if c.isdigit()]
    )
    Nsnaps = len(snapshots)
    print("Found {0} snapshots".format(Nsnaps))

    time = numpy.zeros(Nsnaps) | units.Myr
    com = numpy.zeros(Nsnaps) | units.parsec
    comvel = numpy.zeros(Nsnaps) | units.km/units.s
    Mtot = numpy.zeros(Nsnaps) | units.MSun
    Ekin = numpy.zeros(Nsnaps) | units.J
    Epot = numpy.zeros(Nsnaps) | units.J
    Ltot = numpy.zeros(Nsnaps) | units.MSun*units.parsec**2/units.Myr
    ptot = numpy.zeros(Nsnaps) | units.MSun*units.parsec/units.Myr
    for i, fname in enumerate(snapshots):
        # if i > 2: break
        print("  Loading snapshot: {0}".format(fname))
        stars = read_set_from_file(fname, "hdf5")
        time[i] = stars.get_timestamp().as_quantity_in(units.Myr)
        print("  This snapshot was saved at T={0}".format(time[i]))
        # (com[i], comvel[i], Mtot[i], Ekin[i], Epot[i], Ltot[i], ptot[i]) = \
        #     print_stuff(stars, w="    ")
        com_i, comvel_i, Mtot_i, Ekin_i, Epot_i, Ltot_i, ptot_i = print_stuff(stars, w="    ")
        # com[i], comvel[i], Mtot[i], Ekin[i], Epot[i], Ltot[i], ptot[i] = \
        #     com_i, comvel_i, Mtot_i, Ekin_i, Epot_i, Ltot_i, ptot_i
        Ekin[i] = Ekin_i
        Epot[i] = Epot_i
        print("")

        isolation = True
        softening=0.1
        fname = "{}{}_{}_{}_{}_{}_{:04d}.png".format(obs.outdir, obs.gc_slug,
            model_name, "isolation" if isolation else "MWPotential2014", len(stars),
            softening, i)
        plot_SigmaR_vs_R(obs, limepy_model, stars,
            model_name=model_name, Tsnap=time[i], softening=softening,
        ).savefig(fname); pyplot.close(pyplot.gcf())

        print("\nSaved: {0}".format(fname))


    # Check energy
    fig, ax = pyplot.subplots(1, 1, figsize=(12, 9))
    ax.plot(time.value_in(units.Myr), Ekin.value_in(units.J), label="Ekin")
    ax.plot(time.value_in(units.Myr), -Epot.value_in(units.J), label="-Epot")
    ax.plot(time.value_in(units.Myr), (Ekin+Epot).value_in(units.J), label="Ekin+Etot")
    ax.set_xlabel("Time [ Myr ]")
    ax.set_ylabel("Energy [ J ]")
    ax.legend(loc="best", fontsize=16, frameon=False)
    pyplot.show(fig)



def dump_snapshot(obs, sim, stars, time, i):
    fname = "{}{}_{}_isolation_{}_{}_{:04d}.h5".format(obs.outdir, obs.gc_slug,
        sim.model_name, sim.Nstars, sim.softening.value_in(units.parsec), i
    )
    print("\n    Dumping snapshot: {0}".format(fname))
    if os.path.exists(fname) and os.path.isfile(fname):
        print("      WARNING: file exists, overwriting it!")

    # append_to_file --> existing file is removed and overwritten
    write_set_to_file(stars.copy_to_new_particles().savepoint(time),
        fname, "hdf5", append_to_file=False)
    print("    done")

    return stars


def new_argument_parser():
    args = argparse.ArgumentParser(
        description="Integrate Nbody realisation of Globular Cluster in isolation."
    )

    args.add_argument("-gc", "--gc_name", dest="gc_name", default="NGC 104",
        type=str, help="Name of the Globular Cluster."
    )
    args.add_argument("-m", "--model_name", dest="model_name", default="king", type=str,
        choices=["king", "wilson", "limepy"],
        help="Physical model for the density structure of the Globular Cluster.",
    )
    args.add_argument("-N", "--Nstars", dest="Nstars", default=1000, type=int,
        help="Number of particles.",
    )
    args.add_argument("-t", "--end-time", dest="endtime", default=1000.0,
        type=float, help="obsulation end time. Use float, in Myr!",
    )
    args.add_argument("--Nsnap", dest="Nsnapshots", default=100, type=int,
        help="Number of snapshots to save",
    )
    args.add_argument("-c", "--code", dest="code", default="fi",
        type=str, choices=["fi", "ph4", "huayno", "bhtree", "gadget2",
            "mercury", "hermite", "phigrape"],
        help="Nbody integrator to use for the obsulation.",
    )
    args.add_argument("-np", "--number_of_workers", dest="number_of_workers",
        default=4, type=int, help="Number of workers",
    )
    args.add_argument("-s", "--softening", dest="softening", default=0.1,
        type=float, help="Force softening. Use float, in pc!",
    )
    args.add_argument("--seed", dest="seed", default=-1, type=int,
        help="Random number seed (-1, no seed).",
    )

    return args


if __name__ == "__main__":
    import logging
    logging.getLogger("keyring").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(message)s")
    logger = logging.getLogger(__file__)

    args, unknown = new_argument_parser().parse_known_args()

    logger.info("Running {0}".format(__file__))
    logger.info("  gc_name: {0}".format(args.gc_name))
    logger.info("  model_name: {0}".format(args.model_name))
    logger.info("  Nstars: {0}".format(args.Nstars))
    logger.info("  endtime: {0} [Myr]".format(args.endtime))
    logger.info("  Nsnapshots: {0}".format(args.Nsnapshots))
    logger.info("  code: {0}".format(args.code))
    logger.info("  number_of_workers: {0}".format(args.number_of_workers))
    logger.info("  softening: {0} [parsec]".format(args.softening))
    logger.info("  seed: {0}\n".format(args.seed))

    if "/tidalshocks" not in sys.path:
        sys.path.insert(0, "{}/tidalshocks/src".format(BASEDIR))
    from mw_gc_observation import MwGcObservation
    from galpy_amuse_wrapper import MwGcSimulation

    obs = MwGcObservation(logger, args.gc_name)
    sim = MwGcSimulation(logger, obs, args.model_name, Nstars=args.Nstars,
        endtime=args.endtime, Nsnapshots=args.Nsnapshots, code=args.code,
        number_of_workers=args.number_of_workers, softening=args.softening,
        isolation=True, seed=args.seed, do_something=dump_snapshot
    )
