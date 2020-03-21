import os
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


def analyse_isolation(obs, sim, rmin=1e-3, rmax=1e3, Nbins=256, smooth=False):
    snap_base = "{}{}_isolation_*.hdf5".format(obs.outdir, obs.gc_name)
    snapshots = sorted(
        glob.glob(snap_base)
    )
    print("Found {0} snapshots".format(len(snapshots)))

    for i, fname in enumerate(snapshots):
        print("  Loading snapshot: {0}".format(fname))
        Tsnap = fname.split("T=")[-1].split("_i")[0] | units.Myr
        stars = read_set_from_file(fname, "hdf5")
        print(stars.total_mass().as_quantity_in(units.MSun))
        print(stars.center_of_mass().as_quantity_in(units.parsec))
        # Tsnap = stars.get_timestamp()  # if stars.savepoint(time) would have been used
        # However, ParticlesWithUnitsConverted does not have savepoint whereas
        # Particles does...
        print("  This snapshot was saved at T={0}".format(Tsnap.as_quantity_in(units.Myr)))
        print("")
        modelname = "King, loaded, T={0} Myr".format(Tsnap.value_in(units.Myr))
        # print_particleset_info(stars, converter, modelname)

        # fig = scatter_particles_xyz(obs.king_amuse)
        # pyplot.show(fig)

        # Mass
        fig, ax = pyplot.subplots(1, 1, figsize=(12, 9))
        obs.add_deBoer2019_sampled_to_ax(ax, stars, model=model,
            parm="mc", rmin=rmin, rmax=rmax, Nbins=Nbins, smooth=smooth)
        pyplot.show(fig)


def dump_snapshot(stars, time, i, model_name):
    fname = "{}{}_{}_isolation_{:04d}.h5".format(obs.outdir, obs.gc_slug, model_name, i)
    print("\n    Dumping snapshot: {0}".format(fname))
    if os.path.exists(fname) and os.path.isfile(fname):
        print("      WARNING: file exists, overwriting it!")

    # append_to_file --> existing file is removed and overwritten
    write_set_to_file(stars, fname, "hdf5", append_to_file=False)
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
        isolation=True, seed=args.seed, do_something=lambda stars, time, i:
            dump_snapshot(stars, time, i, model_name=args.model_name)
    )
