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
from amuse.io import read_set_from_file


BASEDIR = "/u/timoh/phd" if "freya" in platform.node() else ""


def analyse_isolation(sim, model, rmin=1e-3, rmax=1e3, Nbins=256, smooth=False):
    snap_base = "{}{}_isolation_*.hdf5".format(sim.outdir, sim.gc_name)
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

        # fig = scatter_particles_xyz(sim.king_amuse)
        # pyplot.show(fig)

        # Plot Sigma(R) vs R
        fig, ax = pyplot.subplots(1, 1, figsize=(12, 12))
        sim.add_deBoer2019_to_fig(fig, show_King=True)
        fig.suptitle("{0} at T={1} Myr".format(sim.gc_name,
            Tsnap.value_in(units.Myr)), fontsize=22)
        # sim.add_deBoer2019_sampled_to_ax(ax, stars, model=model,
        #     parm="rho", rmin=rmin, rmax=rmax, Nbins=Nbins, smooth=smooth)
        sim.add_deBoer2019_sampled_to_ax(ax, stars, model=model,
            parm="Sigma", rmin=rmin, rmax=rmax, Nbins=Nbins, smooth=smooth)
        ax.legend(fontsize=20)
        pyplot.savefig("{0}{1}_isolation_{2:04d}.png".format(sim.outdir, sim.gc_name, i))
        pyplot.show(fig)

        # Mass
        fig, ax = pyplot.subplots(1, 1, figsize=(12, 9))
        sim.add_deBoer2019_sampled_to_ax(ax, stars, model=model,
            parm="mc", rmin=rmin, rmax=rmax, Nbins=Nbins, smooth=smooth)
        pyplot.show(fig)


class IsolationTestSimulation(object):
    def __init__(self,
                 nstars=1000,
                 endtime=1000,
                 star_code='hermite',
                 softening=0.0,
                 seed=-1,
                 ntimesteps=10,
                 **ignored_options
                 ):
        if seed >= 0:
            numpy.random.seed(seed)


def new_argument_parser():
    args = argparse.ArgumentParser(
        description="Integrate Nbody realisation of Globular Cluster in isolation."
    )

    args.add_argument("-gc", "--gc_name", dest="gc_name", default="NGC 104",
        type=str, help="Name of the Globular Cluster."
    )
    args.add_argument("-m", "--model", dest="model", default="king", type=str,
        choices=["king", "wilson", "limepy"],
        help="Physical model for the density structure of the Globular Cluster.",
    )
    args.add_argument("-N", "--Nstars", dest="Nstars", default=1000, type=int,
        help="Number of particles.",
    )
    args.add_argument("-c", "--code", dest="code", default="hermite",
        type=str, choices=["hermite", "bhtree", "octgrav", "phigrape", "ph4"],
        help="Nbody integrator to use for the simulation.",
    )
    args.add_argument("-t", "--end-time", dest="endtime", default=1000.0,
        type=float, help="Simulation end time. Use float, in Myr!",
    )
    args.add_argument("--nsnap", dest="nsnap", default=100, type=int,
        help="Number of snapshots to save",
    )
    args.add_argument("-s", "--softening", dest="softening", default=0.1,
        type=float, help="Force softening. Use float, in pc!",
    )
    args.add_argument("--seed", dest="seed", default=0, type=int,
        help="Random number seed (-1, no seed).",
    )
    args.add_argument("-np", "--number_of_workers", dest="number_of_workers",
        default=4, type=int, help="Number of workers",
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
    logger.info("  model: {0}".format(args.model))
    logger.info("  Nstars: {0}".format(args.Nstars))
    logger.info("  code: {0}".format(args.code))
    logger.info("  endtime: {0} [Myr]".format(args.endtime))
    logger.info("  nsnap: {0}".format(args.nsnap))
    logger.info("  softening: {0} [parsec]".format(args.softening))
    logger.info("  seed: {0}".format(args.seed))
    logger.info("  number_of_workers: {0}\n".format(args.number_of_workers))

    if "/tidalshocks" not in sys.path:
        sys.path.insert(0, "{}/tidalshocks/src".format(BASEDIR))
    from mw_gc_observation import MwGcObservation

    sim = StarClusterSimulation(logger, args.gc_name)

    # Sample initial conditions for King/Wilson/Limepy MLEs from deBoer+ 2019
    if args.model == "king":
        model, limepy_sampled, amuse_sampled, converter = \
            sim.sample_deBoer2019_bestfit_king(Nstars=args.Nstars)
    elif args.model == "wilson":
        model, limepy_sampled, amuse_sampled, converter = \
            sim.sample_deBoer2019_bestfit_wilson(Nstars=args.Nstars)
    elif args.model == "limepy":
        model, limepy_sampled, amuse_sampled, converter = \
            sim.sample_deBoer2019_bestfit_limepy(Nstars=args.Nstars)

    # Verify that the sampled profile matches the observed profile (as well as
    # the requested Limepy model, of course)

    # Plot the initial conditions as a sanity check
    # fig, ax = pyplot.subplots(1, 1, figsize=(12, 9))
    # sim.add_deBoer2019_to_fig(fig, show_King=True)
    # sim.add_deBoer2019_sampled_to_ax(ax, king_amuse_sampled, parm="Sigma",
    #     model=king_model, rmin=1e-4, rmax=1e3, Nbins=int(numpy.sqrt(args.Nstars)))
    # ax.legend(fontsize=20)
    # fname = "{0}{1}_sampled_ICs.png".format(sim.outdir, sim.gc_name)
    # pyplot.savefig(fname)
    # pyplot.close(fig)
    # logger.info("  Saved: {0}".format(fname))

    # # Run the simulation
    # gc_in_isolation(sim, king_amuse_sampled, king_converter, ts=ts, tsnap=tsnap,
    #     softening=softening, number_of_workers=args.number_of_workers)
