import os
import numpy
from matplotlib import pyplot
from matplotlib import gridspec
from amuse.units import units
from amuse.units import nbody_system
from amuse.ic.plummer import new_plummer_sphere
from amuse.community.bhtree.interface import BHTree
from amuse.community.gadget2.interface import Gadget2
from amuse.couple import bridge
from galpy.potential import to_amuse
from galpy.potential import MWPotential2014


def setup_cluster(
        N=1000,
        Mcluster=1000.0 | units.MSun,
        Rcluster=10.0 | units.parsec,
        Rinit=[10.0, 0.0, 0.0] | units.kpc,
        Vinit=[0.0, 220.0, 0.0] | units.km / units.s):
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


def integrate_Nbody_in_MWPotential2014_with_AMUSE(
        stars, converter, tend=100.0 | units.Myr, dt=1.0 | units.Myr,
        softening=3.0 | units.parsec, opening_angle=0.6,
        number_of_workers=1, do_something=lambda stars, time, i: x,
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
    time = 0.0 | tend.unit
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
        break

    channel_from_gravity_to_stars.copy()
    gravity.stop()


def plot_stars(stars, time, i):
    fig, ax = pyplot.subplots(figsize=(12, 12))
    ax.plot(stars.x.value_in(units.kpc), stars.y.value_in(units.kpc), "ko", ms=2)

    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")

    fig.savefig("{0}/out/test_{1:03d}.png".format(
        "/".join(os.path.abspath(__file__).split("/")[:-2]), i))
    pyplot.close(fig)


def plot_center_of_mass(stars, time, i):
    fig = pyplot.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(3, 3)
    axxz = fig.add_subplot(gs.new_subplotspec((0, 0), colspan=2))
    axzy = fig.add_subplot(gs.new_subplotspec((1, 2), rowspan=2))
    axxy = fig.add_subplot(gs.new_subplotspec((1, 0), colspan=2, rowspan=2))
                              #sharex=axxz, sharey=axzy)
    
    axt = fig.add_subplot(gs.new_subplotspec((0, 2)))
    axt.axis("off")
     
    axxy.plot(CoM_array[:,0]/1000, CoM_array[:,1]/1000)
    axxy.set_xlabel("x [kpc]")
    axxy.set_ylabel("y [kpc]")
    
    axzy.plot(CoM_array[:,2]/1000, CoM_array[:,1]/1000)
    axzy.set_xlabel("z [kpc]")
    axzy.set_ylabel("y [kpc]")
    
    axxz.plot(CoM_array[:,0]/1000, CoM_array[:,2]/1000)
    axxz.set_xlabel("x [kpc]")
    axxz.set_ylabel("z [kpc]")
    
    fig.subplots_adjust(wspace=0, hspace=0, left=0.09, right=0.98, bottom=0.07, top=0.98)
    pyplot.show(fig)


if __name__ == "__main__":
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

    integrate_Nbody_in_MWPotential2014_with_AMUSE(
        stars, converter, do_something=calculate_com_and_plot
    )

    print(com)
