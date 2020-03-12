import os
import time
import numpy
import limepy
import logging
import astropy.units as u
from matplotlib import pyplot
from matplotlib import gridspec
from amuse.units import units
from amuse.units import nbody_system
from amuse.datamodel import Particles
from amuse.ic.plummer import new_plummer_sphere
from amuse.community.bhtree.interface import BHTree
from amuse.community.gadget2.interface import Gadget2
from amuse.couple import bridge
from galpy.orbit import Orbit
from galpy.potential import to_amuse
from galpy.potential import MWPotential2014


def limepy_to_amuse(W0, M=1e5, rt=3, g=1, N=1000, seed=1337, verbose=False):
    model = limepy.limepy(W0, M=M, rt=rt, g=g, verbose=verbose)
    particles = limepy.sample(model, N=N, seed=seed, verbose=verbose)
    amuse = Particles(size=N)
    amuse.x = (particles.x) | units.parsec
    amuse.y = (particles.y) | units.parsec
    amuse.z = (particles.z) | units.parsec
    amuse.vx = particles.vx | units.km/units.s
    amuse.vy = particles.vy | units.km/units.s
    amuse.vz = particles.vz | units.km/units.s
    amuse.mass = particles.m | units.MSun
    return model, particles, amuse


def setup_cluster(
        N=1000,
        Mcluster=1000.0 | units.MSun,
        Rcluster=10.0 | units.parsec,
        Rinit=[10.0, 0.0, 0.0] | units.kpc,  # x, y, z in kpc
        Vinit=[0.0, 220.0, 0.0] | units.km / units.s):  # vx, vy, vz in km / s
    """ Setup an Nbody star cluster at a given location within the Galaxy """

    converter = nbody_system.nbody_to_si(Mcluster, Rcluster)
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
        dt=1.0 | units.Myr, softening=3.0 | units.parsec, opening_angle=0.6,
        number_of_workers=1, do_something=lambda stars, time, i: stars,
    ):
    """ Integrate an Nbody star cluster in the MWPotential2014 """

    # Convert galpy MWPotential2014 to AMUSE representation (Webb+ 2020)
    mwp_amuse = to_amuse(MWPotential2014)

    cluster_code = BHTree(converter, number_of_workers=number_of_workers)
    cluster_code.parameters.epsilon_squared = (softening)**2
    cluster_code.parameters.opening_angle = opening_angle
    cluster_code.parameters.timestep = dt
    cluster_code.particles.add_particles(stars)

    # Setup channels between stars particle dataset and the cluster code
    channel_from_stars_to_gravity = stars.new_channel_to(
        cluster_code.particles, attributes =["mass", "x", "y", "z", "vx", "vy", "vz"])
    channel_from_gravity_to_stars = cluster_code.particles.new_channel_to(
        stars, attributes =["mass", "x", "y", "z", "vx", "vy", "vz"])

    # Setup gravity bridge
    gravity = bridge.Bridge(use_threading=False)
    # Stars in gravity depend on gravity from external potential mwp_amuse (i.e., MWPotential2014)
    gravity.add_system(cluster_code, (mwp_amuse, ))
    # External potential mwp_amuse still needs to be added to system so it evolves with time
    gravity.add_system(mwp_amuse, )
    # Set how often to update external potential
    gravity.timestep = cluster_code.parameters.timestep / 2.0

    Nsteps = int(tend/dt)
    time = tstart
    try:
        while time < tend:
            i = int(time/tend * Nsteps)

            # Evolve
            gravity.evolve_model( time+dt )

            # Copy stars from gravity to output or analyze the simulation
            channel_from_gravity_to_stars.copy()

            stars = do_something(stars, time, i)

            # If you edited the stars particle set, for example to remove stars from the
            # array because they have been kicked far from the cluster, you need to
            # copy the array back to gravity:
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


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(message)s")
    logger = logging.getLogger(__file__)
    logger.info("Running {0}".format(__file__))

    stars, converter = setup_cluster()

    tend = 100.0 | units.Myr
    dt = 1.0 | units.Myr
    Nsteps = int(tend/dt)

    com = numpy.zeros((Nsteps,3))
    times = numpy.zeros(Nsteps)
    def calculate_com_and_plot(stars, time, i):
        c = stars.center_of_mass().value_in(units.parsec)
        com[i][:] = c
        times[i] = time.value_in(units.Myr)
        plot_stars(stars, time, i)
        return stars

    integrate_Nbody_in_MWPotential2014_with_AMUSE(logger,
        stars, converter, do_something=calculate_com_and_plot
    )

    logger.info(com)
