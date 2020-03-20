import os
import sys
import time
import numpy
import platform
import argparse
import astropy.units as u
from matplotlib import pyplot
from matplotlib import gridspec
from amuse.units import units
from amuse.units import nbody_system
from amuse.io import write_set_to_file
from amuse.io import read_set_from_file
from amuse.datamodel import Particles
from amuse.datamodel import ParticlesWithUnitsConverted
from amuse.ic.plummer import new_plummer_sphere
from amuse.community.fi.interface import Fi
from amuse.community.bhtree.interface import BHTree
from amuse.community.gadget2.interface import Gadget2
from amuse.couple import bridge
from galpy.orbit import Orbit
from galpy.potential import to_amuse
from galpy.potential import MWPotential2014


BASEDIR = "/u/timoh/phd" if "freya" in platform.node() else ""
if "/limepy" not in sys.path:
    sys.path.insert(0, "{}/limepy".format(BASEDIR))
import limepy   # using tlrh314/limepy fork


def limepy_to_amuse(W0, M=1e5, rt=3.0, g=1, Nstars=1000, seed=1337,
        Rinit=[0.0, 0.0, 0.0] | units.kpc,  # x, y, z in kpc
        Vinit=[0.0, 0.0, 0.0] | units.km / units.s,  # vx, vy, vz in km / s
        verbose=False, timing=True):

    start = time.time()

    # Setup the Limepy model - CAUTION: do **not** scale here b/c we use the
    # AMUSE converter later to scale the model from Nbody to physical units.
    model = limepy.limepy(W0, g=g, M=1, rt=1, G=1, verbose=verbose)
    if timing:
        print("limepy.limepy took {0:.2f} s".format(time.time() - start))

    # Sample the (unscaled) model using Limepy's built-in sample routine
    start = time.time()
    particles = limepy.sample(model, N=Nstars, seed=seed, verbose=verbose)
    if timing:
        print("limepy.sample took {0:.2f} s".format(time.time() - start))

    # Move the Limepy particles into an AMUSE datamodel Particles instance
    # Note that we use nbody units here and the AMUSE converter to scale afterwards
    start = time.time()
    amuse = Particles(size=Nstars)
    amuse.x = particles.x | nbody_system.length
    amuse.y = particles.y | nbody_system.length
    amuse.z = particles.z | nbody_system.length
    amuse.vx = particles.vx | nbody_system.length / nbody_system.time
    amuse.vy = particles.vy | nbody_system.length / nbody_system.time
    amuse.vz = particles.vz | nbody_system.length / nbody_system.time
    amuse.mass = particles.m | nbody_system.mass

    # Move particles to center of mass
    amuse.move_to_center()

    if timing:
        print("convert to AMUSE took {0:.2f} s".format(time.time() - start))

    print("limepy_to_amuse, pre-converter business")
    print("  G:      {}".format(nbody_system.G))
    print("  com:    {}".format(amuse.center_of_mass()))
    print("  comvel: {}".format(amuse.center_of_mass_velocity()))
    Mtot = amuse.total_mass()
    print("  Mtot:   {}".format(Mtot))
    Ekin = amuse.kinetic_energy()
    print("  Ekin:   {}".format(Ekin))
    Epot = amuse.potential_energy(G=nbody_system.G)
    print("  Epot:   {}".format(Epot))
    print("  Ekin/Epot: {}".format(Ekin/Epot))
    Ltot = amuse.total_angular_momentum()
    print("  Ltot:   {}".format(Ltot))
    ptot = amuse.total_momentum()
    print("  ptot:   {}".format(ptot))

    # Setup the converter to pass to the AMUSE Nbody code
    converter = nbody_system.nbody_to_si(M | units.MSun, rt | units.parsec)

    # And apply the converter to scale the Nbody units to physical units. The
    # converter can later be passed to the Nbody solver and we no longer need
    # to worry about the physical units as that will be handled by AMUSE :-)
    amuse = ParticlesWithUnitsConverted(amuse, converter.as_converter_from_si_to_nbody())

    # Finally, add the initial position and velocity vectors of the GC within the Galaxy
    amuse.x += Rinit[0]
    amuse.y += Rinit[1]
    amuse.z += Rinit[2]
    amuse.vx += Vinit[0]
    amuse.vy += Vinit[1]
    amuse.vz += Vinit[2]

    print("limepy_to_amuse, post-converter business")
    print("  com:    {}".format(amuse.center_of_mass().as_quantity_in(units.parsec)))
    print("  comvel: {}".format(amuse.center_of_mass_velocity().as_quantity_in(units.km/units.s)))
    Mtot = amuse.total_mass().as_quantity_in(units.MSun)
    print("  Mtot:   {}".format(Mtot))
    Ekin = amuse.kinetic_energy().as_quantity_in(units.J)
    print("  Ekin:   {}".format(Ekin))
    Epot = amuse.potential_energy().as_quantity_in(units.J)
    print("  Epot:   {}".format(Epot))
    print("  Ekin/Epot: {}".format(Ekin/Epot))
    Ltot = amuse.total_angular_momentum().as_quantity_in(units.MSun*units.parsec**2/units.Myr)
    print("  Ltot:   {}".format(Ltot))
    ptot = amuse.total_momentum().as_quantity_in(units.MSun*units.parsec/units.Myr)
    print("  ptot:   {}".format(ptot))

    # Now let limepy scale the model because that's what we'll return
    # and we kinda like the model to be in physical units. Also project :-).
    model = limepy.limepy(W0, g=g, M=M, rt=rt, G=1, verbose=verbose, project=True)
    return model, particles, amuse, converter


def setup_cluster(N=1000, Mcluster=1000.0 | units.MSun, Rcluster=10.0 | units.parsec,
        Rinit=[10.0, 0.0, 0.0] | units.kpc,  # x, y, z in kpc
        Vinit=[0.0, 220.0, 0.0] | units.km / units.s):  # vx, vy, vz in km / s
    """ Setup an Nbody star cluster at a given location within the Galaxy """

    converter = nbody_system.nbody_to_si(Mcluster, Rcluster)
    print("\nsetup_cluster converter units\n{}\n\n".format(converter.units))
    stars = new_plummer_sphere(N, converter)
    stars.x += Rinit[0]
    stars.y += Rinit[1]
    stars.z += Rinit[2]
    stars.vx += Vinit[0]
    stars.vy += Vinit[1]
    stars.vz += Vinit[2]

    return stars, converter


def integrate_Nbody_in_MWPotential2014_with_AMUSE(logger,
        stars, converter, tstart=0.0 | units.Myr, tend=100.0 | units.Myr,
        dt=1.0 | units.Myr, softening=0.1 | units.parsec, opening_angle=0.6,
        number_of_workers=1, do_something=lambda stars, time, i: stars,
        run_in_isolation=False,
    ):
    """ Integrate an Nbody star cluster in the MWPotential2014 """

    print("Setting up cluster_code /w {} workers".format(number_of_workers))
    print("converter: {}".format(converter))
    # cluster_code = BHTree(converter, number_of_workers=number_of_workers)
    os.environ["OMP_NUM_THREADS"] = "4"
    cluster_code = Fi(converter, number_of_workers=1, mode="openmp")
    cluster_code.parameters.epsilon_squared = (softening)**2
    cluster_code.parameters.opening_angle = opening_angle
    cluster_code.parameters.timestep = dt
    print(cluster_code.parameters.timestep)
    print(cluster_code.parameters)
    cluster_code.particles.add_particles(stars)
    print("Done setting up cluster_code")

    # Setup channels between stars particle dataset and the cluster code
    print("Setting up channel from stars to gravity and vice versa")
    channel_from_stars_to_gravity = stars.new_channel_to(
        cluster_code.particles, attributes =["mass", "x", "y", "z", "vx", "vy", "vz"])
    channel_from_gravity_to_stars = cluster_code.particles.new_channel_to(
        stars, attributes =["mass", "x", "y", "z", "vx", "vy", "vz"])
    print("Done setting up channels")

    # Setup gravity bridge
    print("Setting up bridge")
    gravity = bridge.Bridge(use_threading=False)
    if run_in_isolation:
        # As a stability check, we first simulate the star cluster in isolation
        gravity.add_system(cluster_code, )
    else:
        # Convert galpy MWPotential2014 to AMUSE representation (Webb+ 2020)
        mwp_amuse = to_amuse(MWPotential2014)
        # Stars in gravity depend on gravity from external potential mwp_amuse (i.e., MWPotential2014)
        gravity.add_system(cluster_code, (mwp_amuse, ))
        # External potential mwp_amuse still needs to be added to system so it evolves with time
        gravity.add_system(mwp_amuse, )
    # Set how often to update external potential
    gravity.timestep = cluster_code.parameters.timestep / 2.0
    print("Done setting up bridge")

    Nsteps = int(tend/dt)
    time = tstart
    print("Nsteps, time, dt, tsnap = {}, {}, {}, {}".format(
        Nsteps, time, dt.value_in(units.Myr), tsnap.value_in(units.Myr)
    ))

    # Also do_something for the ICs
    i = int(time/tend * Nsteps)  # should be 0
    stars = do_something(stars, time, i)
    try:
        while time < tend:
            # Evolve
            print("gravity.evolve_model", time.as_quantity_in(units.Myr))
            gravity.evolve_model( time+dt )
            if (time+dt).value_in(units.Myr) % tsnap.value_in(units.Myr) < 1e-9:
                print("SNAP")

                # Copy stars from gravity to output or analyze the simulation
                print("channel_from_gravity_to_stars.copy()")
                channel_from_gravity_to_stars.copy()

                i = int((time+dt)/tend * Nsteps)
                stars = do_something(stars, time+dt, i)

                # If you edited the stars particle set, for example to remove stars from the
                # array because they have been kicked far from the cluster, you need to
                # copy the array back to gravity:
                print("channel_from_stars_to_gravity.copy()")
                channel_from_stars_to_gravity.copy()

            # Update time
            time = gravity.model_time
            # break

        channel_from_gravity_to_stars.copy()
        gravity.stop()
    except Exception as e:
        logger.error(e)
        gravity.stop()
        raise


def plot_stars(stars, time, i, fname="test_"):
    fig, ax = pyplot.subplots(figsize=(12, 12))
    ax.plot(stars.x.value_in(units.kpc), stars.y.value_in(units.kpc), "ko", ms=2)

    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")

    fig.savefig("{0}/out/{1}{2:03d}.png".format(
        "/".join(os.path.abspath(__file__).split("/")[:-2]), fname, i))
    pyplot.close(fig)


def plot_center_of_mass(com, fig=None):
    if fig is None:
        fig = pyplot.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(3, 3)
    axxz = fig.add_subplot(gs.new_subplotspec((0, 0), colspan=2))
    axzy = fig.add_subplot(gs.new_subplotspec((1, 2), rowspan=2))
    axxy = fig.add_subplot(gs.new_subplotspec((1, 0), colspan=2, rowspan=2))
                              #sharex=axxz, sharey=axzy)

    axt = fig.add_subplot(gs.new_subplotspec((0, 2)))
    axt.axis("off")

    axxy.plot(com[:,0]/1000, com[:,1]/1000, "ko", ms=4)
    axxy.set_xlabel("x [kpc]")
    axxy.set_ylabel("y [kpc]")

    axzy.plot(com[:,2]/1000, com[:,1]/1000, "ko", ms=4)
    axzy.set_xlabel("z [kpc]")
    axzy.set_ylabel("y [kpc]")

    axxz.plot(com[:,0]/1000, com[:,2]/1000, "ko", ms=4)
    axxz.set_xlabel("x [kpc]")
    axxz.set_ylabel("z [kpc]")

    fig.subplots_adjust(wspace=0, hspace=0, left=0.09, right=0.98, bottom=0.07, top=0.98)
    return fig


def plot_galpy_and_amuse_integrations(o, com, ts, gc_name):
    tstart, tend, Nsteps, dt = convert_galpy_to_amuse_times(ts)

    # AMUSE integration
    fig = plot_center_of_mass(com)
    axxz, axzy, axxy, axt = fig.axes

    # Initial position
    x0, y0, z0 = com[:,0][0]/1000, com[:,1][0]/1000, com[:,2][0]/1000
    axxz.plot(x0, z0, "rX", ms=10)
    axzy.plot(z0, y0, "rX", ms=10)
    axxy.plot(x0, y0, "rX", ms=10)

    # Final position
    axxz.plot(com[-1:,0]/1000, com[-1:,2]/1000, "ro", ms=10)
    axzy.plot(com[-1:,2]/1000, com[-1:,1]/1000, "ro", ms=10)
    axxy.plot(com[-1:,0]/1000, com[-1:,1]/1000, "ro", ms=10)

    # Galpy integration
    pyplot.sca(axxz); o.plot(d1="x", d2="z", overplot=True, use_physical=True, c="k", ls=":")
    pyplot.sca(axzy); o.plot(d1="z", d2="y", overplot=True, use_physical=True, c="k", ls=":")
    pyplot.sca(axxy); o.plot(d1="x", d2="y", overplot=True, use_physical=True, c="k", ls=":")

    timestring = "T = {0:.2f}-{1:.2f} Myr".format(tstart.value_in(units.Myr), tend.value_in(units.Myr))
    dtstring = "dt = {0:.8f} Myr".format(dt.value_in(units.Myr))
    startstring = "$\\vec{{x_0}}$ = {:.1f}, {:.1f}, {:.1f} kpc".format(x0, y0, z0)
    axt.text(0.5, 0.95, gc_name, fontsize=16, ha="center", va="center", transform=axt.transAxes)
    axt.text(0.5, 0.85, timestring, fontsize=16, ha="center", va="center", transform=axt.transAxes)
    axt.text(0.5, 0.75, dtstring, fontsize=16, ha="center", va="center", transform=axt.transAxes)
    axt.text(0.5, 0.65, startstring, fontsize=16, ha="center", va="center", transform=axt.transAxes)

    pyplot.show(fig)


def convert_galpy_to_amuse_times(ts):
    tstart = ts[0].value | units.Gyr
    tend = ts[-1].value | units.Gyr
    Nsteps = len(ts)
    dt = (tend - tstart) / Nsteps

    return tstart, tend, Nsteps, dt


def compare_galpy_and_amuse(logger, h19_o, h19_combined, N=1,
        ts=numpy.linspace(0.0, 1, 8096+1) * u.Gyr,
        do_something=lambda stars, time, i: stars,
        number_of_workers=1,
    ):

    gc_name = h19_o["Cluster"]
    tstart, tend, Nsteps, dt = convert_galpy_to_amuse_times(ts)

    # For Galpy.Orbit.integrate --> astropy units
    logger.info("Integrating Galpy.Orbit from T={0:.1f} to T={1:.1f} in N={2} steps".format(ts[0], ts[-1], len(ts)))

    # For AMUSE integrations --> AMUSE units
    logger.info("Integrating AMUSE orbit from T={0:.1f} Gyr to T={1:.1f} Gyr in N={2} steps, dt={3} Myr".format(
        tstart.value_in(units.Gyr), tend.value_in(units.Gyr), Nsteps, dt.value_in(units.Myr)))

    logger.info("-"*101)
    logger.info("{:<15s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}|   {:<10s}{:<10s}".format(
        "gc_name", "RA", "Dec", "R_Sun", "v_r", "pmRA", "pmDec", "R_peri", "R_apo"
    ))
    logger.info("{:<15s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}|   {:<10s}{:<10s}".format(
        "", "deg", "deg", "kpc", "km/s", "mas/yr", "mas/yr", "", ""
    ))
    logger.info("-"*101)


    # Get the relevant orbital parameters from Hilker+ 2019
    gc_name = h19_o["Cluster"]
    RA = float(h19_o["RA"])
    Dec = float(h19_o["DEC"])
    R_Sun = float(h19_o["Rsun"])
    v_r = float(h19_o["RV"])
    pmRA = float(h19_o["mualpha"])
    pmDec = float(h19_o["mu_delta"])
    R_peri = float(h19_o["RPERI"])
    R_apo = float(h19_o["RAP"])
    logger.info("{:<15s}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}|   {:<10.2f}{:<10.2f}".format(
        gc_name, RA, Dec, R_Sun, v_r, pmRA, pmDec, R_peri, R_apo
    ))

    # Initialise an Orbit instance in galpy
    o = Orbit([
        RA*u.deg, Dec*u.deg, R_Sun*u.kpc,  # RA (deg), DEC (deg), d (kpc)
        pmRA*u.mas/u.year, pmDec*u.mas/u.year,  # mu_ra (mas/yr), mu_dec (mas/yr)
        v_r*u.km/u.s],  # radial velocity (km/s)
        radec=True, uvw=False, lb=False  # explicit tell Orbit init which input method we use
    )
    # Let Galpy calculate the initial conditions for AMUSE  prior to integration
    x, y, z = o.x(), o.y(), o.z()  # kpc
    vx, vy, vz = o.vx(), o.vy(), o.vz()  # km/s

    # Integrate the Orbit in MWPotential2014 /w Galpy's built-in orbit integrator
    start = time.time()
    o.integrate(ts, MWPotential2014)
    runtime = time.time() - start
    logger.info("Runtime Galpy Orbit integration: {:.2f} seconds".format(runtime))
    logger.info("{:<15s}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}|   {:<10.2f}{:<10.2f}".format(
        "  Orbit", o.ra(), o.dec(), o.dist(), v_r, o.pmra(), o.pmdec(), o.rperi(), o.rap()
    ))

    # Integrate the Orbit in to_amuse /w Gadget2 in MWPotential2014
    # Here we take a single particle with the mass of the GC and the same
    # initial conditions to ensure that the Nbody in MWPotential2014 later on
    # will have the correct orbit.
    # TODO: take the abssolute visual magnitude M_v from Harris (1996, 2010 ed.),
    # assume a mass-to-light ratio, and calculate the mass (instead of dynamical mass from h19)
    Mcluster = h19_combined["Mass"] | units.MSun
    # TODO: this is a bullshit number for the radius of the cluster, so update that
    Rcluster = h19_combined["rt"] | units.parsec
    stars, converter = setup_cluster(
        N=N, Mcluster=Mcluster, Rcluster=Rcluster,
        Rinit=[x, y, z] | units.kpc,
        Vinit=[vx, vy, vz] | units.km / units.s
    )

    start = time.time()
    integrate_Nbody_in_MWPotential2014_with_AMUSE(logger,
        stars, converter, tstart=tstart, tend=tend, dt=dt,
        do_something=do_something, number_of_workers=number_of_workers,
    )
    runtime = time.time() - start
    logger.info("Runtime AMUSE Orbit integration: {:.2f} seconds".format(runtime))

    # plot_galpy_and_amuse_integrations(o, com, tstart, tend, dt, gc_name)

    logger.info("-"*101)

    return o


def gc_in_isolation(obs, ICs, converter, ts=numpy.linspace(0.0, 1.0, 1000) * u.Gyr,
        tsnap=0.1|units.Myr, softening=0.1|units.parsec, number_of_workers=1):
    """ Integrate the GC without external potential in AMUSE /w Gadget2 """

    tstart, tend, Nsteps, dt = convert_galpy_to_amuse_times(ts)
    logger.info("  Galpy T={0:.1f} to T={1:.1f} in N={2} steps".format(ts[0], ts[-1], len(ts)))
    logger.info("  AMUSE T={0:.1f} Gyr to T={1:.1f} Gyr in N={2} steps, dt={3} Myr\n".format(
        tstart.value_in(units.Gyr), tend.value_in(units.Gyr), Nsteps, dt.value_in(units.Myr)))

    def dump_snapshot(stars, time, i):
        fname = "{}{}_isolation_T={:010.2f}_i={:04d}.hdf5".format(obs.outdir, obs.gc_name,
            time.value_in(units.Myr), i)
        print("Dumping snapshot: {0}".format(fname))
        if os.path.exists(fname) and os.path.isfile(fname):
            print("WARNING: file exists, overwriting it!")

        modelname = "King, T={0:.2f} Myr".format(time.value_in(units.Myr))
        # append_to_file --> existing file is removed and overwritten
        write_set_to_file(stars, fname, "hdf5", append_to_file=False)
        print("Done")

        return stars

    start = time.time()
    integrate_Nbody_in_MWPotential2014_with_AMUSE(logger,
        ICs, converter, tstart=tstart, tend=tend, dt=dt,
        do_something=dump_snapshot, number_of_workers=number_of_workers,
        softening=softening, run_in_isolation=True,
    )
    runtime = time.time() - start
    logger.info("Runtime AMUSE integration: {:.2f} seconds".format(runtime))


def new_argument_parser():
    args = argparse.ArgumentParser(description=
        "Integrate Nbody realisation of GC on orbit in MWPotential2014")
    args.add_argument("-gc", "--gc_name", dest="gc_name", default="NGC 104",
        type=str, help="Name of the Globular Cluster")

    args.add_argument("-Ns", "--Nstars", dest="Nstars", default=1000,
        type=int, help="Number of particles in Nbody GC representation")
    args.add_argument("-T", "--tend", dest="tend", default=100.0,
        type=float, help="Simulation end time. Use float, in Myr!")
    args.add_argument("-dt", "--timestep", dest="dt", default=0.01,
        type=float, help="Simulation time step. Use float, in Myr!")
    args.add_argument("-tsnap", "--tsnap", dest="tsnap", default=1.0,
        type=float, help="Time between snapshots. Use float, in Myr!")
    args.add_argument("-s", "--softening", dest="softening", default=0.1,
        type=float, help="Force softening. Use float, in pc!")
    args.add_argument("-np", "--number_of_workers", dest="number_of_workers", default=4,
        type=int, help="Number of workers")

    return args


if __name__ == "__main__":
    args, unknown = new_argument_parser().parse_known_args()
    Nsteps = int(args.tend/args.dt)
    ts = numpy.linspace(0.0, args.tend/1000, Nsteps) * u.Gyr
    tsnap = args.tsnap | units.Myr
    softening = args.softening | units.parsec

    import logging
    logging.getLogger("keyring").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(message)s")
    logger = logging.getLogger(__file__)
    logger.info("Running {0}".format(__file__))
    logger.info("  gc_name : {0}".format(args.gc_name))
    logger.info("  Nstars: {0}".format(args.Nstars))
    logger.info("  tend: {0}".format(args.tend))
    logger.info("  dt: {0}".format(args.dt))
    logger.info("  tsnap: {0}".format(args.tsnap))
    logger.info("  softening: {0}".format(args.softening))
    logger.info("  number_of_workers: {0}\n".format(args.number_of_workers))

    from mw_gc_observation import MwGcObservation
    obs = MwGcObservation(logger, args.gc_name)
    logger.info(obs)

    # Example usage
    stars, converter = setup_cluster()

    com = numpy.zeros((Nsteps,3))
    times = numpy.zeros(Nsteps)
    def calculate_com_and_plot(stars, time, i):
        c = stars.center_of_mass().value_in(units.parsec)
        com[i][:] = c
        times[i] = time.value_in(units.Myr)
        plot_stars(stars, time, i)
        return stars

    integrate_Nbody_in_MWPotential2014_with_AMUSE(logger,
        stars, converter, softening=softening,
        do_something=calculate_com_and_plot
    )

    logger.info(com)
