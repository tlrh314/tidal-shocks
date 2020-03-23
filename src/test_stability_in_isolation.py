import os
import re
import sys
import glob
import numpy
import argparse
import platform
import multiprocessing
from functools import partial
from matplotlib import pyplot
pyplot.style.use("tlrh")
if "freya" in platform.node(): pyplot.switch_backend("agg")
from amuse.units import units
from amuse.io import write_set_to_file
from amuse.io import read_set_from_file

from amuse_wrapper import get_particle_properties
from galpy_amuse_wrapper import plot_SigmaR_vs_R


BASEDIR = "/u/timoh/phd" if "freya" in platform.node() else ""


def analyse_snapshot(i_fname_tuple, *args, **kwargs):
    i, fname = i_fname_tuple
    obs = kwargs["obs"]
    model_name = kwargs["model_name"]
    limepy_model = kwargs["limepy_model"]
    isolation = kwargs["isolation"]
    softening = kwargs["softening"]

    print("  Loading snapshot {}: {}".format(i, fname))
    stars = read_set_from_file(fname, "hdf5")
    Tsnap = stars.get_timestamp().value_in(units.Myr)
    print("  {} was saved at T={:.2f} Myr".format(fname, Tsnap))

    com, comvel, Mtot, Ekin, Epot, Ltot, ptot = get_particle_properties(stars, w=None)

    plot_fname = "{}{}_{}_{}_{}_{}_{:04d}.png".format(obs.outdir, obs.gc_slug,
        model_name, "isolation" if isolation else "MWPotential2014", len(stars),
        softening, i)
    fig = plot_SigmaR_vs_R(obs, limepy_model, stars,
        model_name=model_name, Tsnap=Tsnap, softening=softening,
    )
    fig.savefig(plot_fname)
    print("  Saved: {0}".format(plot_fname))
    pyplot.close(fig)

    return (Tsnap, com.value_in(units.parsec), comvel.value_in(units.km/units.s),
        Mtot.value_in(units.MSun), Ekin.value_in(units.J), Epot.value_in(units.J),
        Ltot.value_in(units.MSun*units.parsec**2/units.Myr),
        ptot.value_in(units.MSun*units.parsec/units.Myr)
    )


def analyse_isolation(obs, model_name, Nstars, softening,
        rmin=1e-3, rmax=1e3, Nbins=256, smooth=False):
    # REMOVE
    limepy_model, limepy_sampled, amuse_sampled, converter = \
        obs.sample_deBoer2019_bestfit_king(Nstars=Nstars)
    # END REMOVE

    snap_base = "{}{}_{}_isolation_{}_{}_*.h5".format(obs.outdir, obs.gc_slug,
        model_name, Nstars, softening)
    snapshots = sorted(glob.glob(snap_base), key=lambda s:
        [ int(c) for c in re.split('(\d+)', s) if c.isdigit()]
    )
    Nsnaps = len(snapshots)
    print("\nFound {0} snapshots".format(Nsnaps))

    # Analyse all snapshot [in parallel]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        info = numpy.array(
            p.map(partial(analyse_snapshot, obs=obs, model_name=model_name,
                limepy_model=limepy_model, softening=0.1, isolation=True),
            enumerate(snapshots))
        )

    Tsnap = info[:,0]
    com = info[:,1]
    comvel = info[:,2]
    Mtot = info[:,3]
    Ekin = info[:,4]
    Epot = info[:,5]
    Ltot = info[:,6]
    ptot = info[:,7]

    # Check energy
    Ekin0, Epot0 = Ekin[0], Epot[0]
    fig, ax = pyplot.subplots(1, 1, figsize=(12, 9))
    ax.plot(Tsnap, 100*(Ekin-Ekin0)/Ekin0, label="Ekin")
    ax.plot(Tsnap, 100*(Epot-Epot0)/Epot0, label="Epot")
    ax.plot(Tsnap, 100*((Epot+Ekin) - (Epot0+Ekin0))/(Epot0+Ekin0), label="Ekin+Etot")
    ax.axhline(0, ls=":", c="k", lw=1)
    ax.set_xlabel("Time [ Myr ]")
    ax.set_ylabel("Energy difference / Energy")
    ax.legend(loc="best", fontsize=16, frameon=False)
    pyplot.show(fig)


def dump_snapshot(obs, sim, stars, time, i):
    fname = "{}{}_{}_isolation_{}_{}_{:04d}.h5".format(obs.outdir, obs.gc_slug,
        sim.model_name, sim.Nstars, sim.softening.value_in(units.parsec), i
    )
    print("    Dumping snapshot: {0}".format(fname))
    if os.path.exists(fname) and os.path.isfile(fname):
        print("      WARNING: file exists, overwriting it!")

    # append_to_file --> existing file is removed and overwritten
    write_set_to_file(stars.copy_to_new_particles().savepoint(time),
        fname, "hdf5", append_to_file=False)

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
    logging.getLogger("code").setLevel(logging.CRITICAL)
    logging.getLogger("amuse").setLevel(logging.CRITICAL)
    logging.getLogger("amuse.rfi").setLevel(logging.CRITICAL)
    logging.getLogger("amuse.rfi.channel").setLevel(logging.CRITICAL)
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
