import os
import sys
import time
import numpy
import signal
import platform
import argparse
import matplotlib
import astropy.units as u
from matplotlib import pyplot
from matplotlib import gridspec
from amuse.units import units
from amuse.units import constants
from amuse.units import nbody_system
from amuse.datamodel import Particles
from amuse.units import generic_unit_converter
from amuse.datamodel import ParticlesWithUnitsConverted
from amuse.ic.plummer import new_plummer_sphere
from amuse.couple import bridge
from galpy.orbit import Orbit
from galpy.potential import to_amuse
from galpy.potential import MWPotential2014


BASEDIR = "/u/timoh/phd" if "freya" in platform.node() else ""
if "/limepy" not in sys.path:
    sys.path.insert(0, "{}/limepy".format(BASEDIR))
import limepy   # using tlrh314/limepy fork

from amuse_wrapper import get_particle_properties


if "/supaharris" not in sys.path:
    sys.path.insert(0, "{}/supaharris".format(BASEDIR))
from utils import parsec2arcmin


def limepy_to_amuse(W0, M=1e5, rt=3.0, g=1, Nstars=1000, seed=1337,
        Rinit=[0.0, 0.0, 0.0] | units.kpc,  # x, y, z in kpc
        Vinit=[0.0, 0.0, 0.0] | (units.km / units.s),  # vx, vy, vz in km / s
        verbose=False, timing=True):

    start = time.time()

    # Setup a converter to pass to the AMUSE Nbody code
    converter = generic_unit_converter.ConvertBetweenGenericAndSiUnits(
        1 | units.MSun, 1 | units.parsec, 1 | units.Myr,
    )
    G = converter.to_nbody(constants.G).number

    # Setup the Limepy model
    model = limepy.limepy(W0, g=g, M=M, rt=rt, G=G, project=True, verbose=verbose)
    if timing:
        print("limepy.limepy took {0:.2f} s".format(time.time() - start))

    # Nstars = int(M / 0.8)  # Assume all stars have a mass of 0.8 MSun

    # Sample the model using Limepy's built-in sample routine
    start = time.time()
    particles = limepy.sample(model, N=Nstars, seed=seed, verbose=verbose)
    if timing:
        print("limepy.sample took {0:.2f} s".format(time.time() - start))

    # Move the Limepy particles into an AMUSE datamodel Particles instance
    start = time.time()
    amuse = Particles(size=Nstars)
    amuse.x = particles.x | units.parsec
    amuse.y = particles.y | units.parsec
    amuse.z = particles.z | units.parsec
    amuse.vx = particles.vx | (units.km / units.s)
    amuse.vy = particles.vy | (units.km / units.s)
    amuse.vz = particles.vz | (units.km / units.s)
    amuse.mass = particles.m | units.MSun
    # Calculated by Gadget
    amuse.radius = numpy.zeros(len(amuse)) | units.parsec
    amuse.epsilon = numpy.zeros(len(amuse)) | units.parsec
    amuse.ax = numpy.zeros(len(amuse)) | units.km / units.s**2
    amuse.ay = numpy.zeros(len(amuse)) | units.km / units.s**2
    amuse.az = numpy.zeros(len(amuse)) | units.km / units.s**2

    # Move particles to center of mass
    amuse.move_to_center()  # adjusts particles.position and particles.velocity

    if timing:
        print("convert to AMUSE took {0:.2f} s".format(time.time() - start))

    # Finally, add the initial position and velocity vectors of the GC within the Galaxy
    amuse.x += Rinit[0]
    amuse.y += Rinit[1]
    amuse.z += Rinit[2]
    amuse.vx += Vinit[0]
    amuse.vy += Vinit[1]
    amuse.vz += Vinit[2]

    if verbose:
        start = time.time()
        get_particle_properties(amuse)
        if timing:
            print("get_particle_properties took {0:.2f} s".format(time.time() - start))

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


def plot_SigmaR_vs_R(obs, limepy_model, amuse_sampled, model_name=None, Tsnap=None,
        softening=None, rmin=1e-3, rmax=1e3, Nbins=256, smooth=False):

    fig, ax = pyplot.subplots(1, 1, figsize=(12, 12))

    if type(Tsnap) == float or type(Tsnap) == numpy.float64:
        suptitle = "{} at T={:<10.2f} Myr".format(obs.gc_name, Tsnap)
    elif Tsnap == "ICs":
        suptitle = "{} ICs".format(obs.gc_name)
    else:
        suptitle = "{}".format(obs.gc_name)

    # Observation
    obs.add_deBoer2019_to_fig(fig,
        show_King=True if model_name == "king" else False,
        show_Wilson=True if model_name == "wilson" else False,
        show_limepy=True if model_name == "limepy" else False,
        show_spes=True if model_name == "spes" else False,
    )
    fig.suptitle(suptitle, fontsize=22)

    # Sampled Sigma(R) profile
    obs.add_deBoer2019_sampled_to_ax(ax, amuse_sampled, limepy_model=limepy_model,
        parm="Sigma", rmin=rmin, rmax=rmax, Nbins=Nbins, smooth=smooth, timing=True)

    if softening is not None:
        xlim = ax.get_xlim()
        softening = parsec2arcmin(softening, obs.distance_kpc)
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(numpy.arange(xlim[0], softening, 0.01), 0, 1,
            facecolor="grey", edgecolor="grey", alpha=0.2, transform=trans)

    ax.legend(loc="lower left", fontsize=20)
    return fig


def plot_diagnostics(obs, limepy_model, stars, model_name=None, Tsnap=None,
        softening=None, rmin=1e-3, rmax=1e3, Nbins=None, fig=None):

    if fig is None:
        fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2, figsize=(24, 24))
    else:
        ax1, ax2, ax3, ax4 = fig.axes
    Nbins = int(numpy.sqrt(len(stars))) if Nbins is None else Nbins

    if type(Tsnap) == float or type(Tsnap) == numpy.float64:
        suptitle = "{} at T={:<10.2f} Myr".format(obs.gc_name, Tsnap)
    elif Tsnap == "ICs":
        suptitle = "{} ICs".format(obs.gc_name)
    else:
        suptitle = "{}".format(obs.gc_name)
    fig.suptitle(suptitle, fontsize=22)

    # Observed projected star count profile of deBoer+ 2019
    pyplot.sca(ax1)
    obs.add_deBoer2019_to_fig(fig, show_King=True, convert_to_parsec=True)
    obs.add_deBoer2019_sampled_to_ax(ax1, stars, parm="Sigma",
        limepy_model=limepy_model, rmin=rmin, rmax=rmax, Nbins=Nbins)

    # Observed projected velocity dispersion profile of Hilker+ 2019
    pyplot.sca(ax2)
    obs.add_H19_RVs_to_fig(fig)
    obs.add_deBoer2019_sampled_to_ax(ax2, stars, parm="v2p",
        limepy_model=limepy_model, rmin=rmin, rmax=rmax, Nbins=Nbins)
    ax2.set_ylim(0.5, 25)

    # Inferred density profile (not projected)
    obs.add_deBoer2019_sampled_to_ax(ax3, stars, parm="rho",
        limepy_model=limepy_model, rmin=rmin, rmax=rmax, Nbins=Nbins)
    ax3.set_ylim(0.1, 1e5)

    # Mass profile
    obs.add_deBoer2019_sampled_to_ax(ax4, stars, parm="mc",
        limepy_model=limepy_model, rmin=rmin, rmax=rmax, Nbins=Nbins)
    ax4.set_ylim(0.1, 3e5)

    for ax in fig.axes:
        ax.set_xlim(rmin, rmax)

        if softening is not None:
            xlim = ax.get_xlim()
            softening = parsec2arcmin(softening, obs.distance_kpc)
            trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.fill_between(numpy.arange(xlim[0], softening, 0.01), 0, 1,
                facecolor="grey", edgecolor="grey", alpha=0.2, transform=trans)

        ax.legend(loc="lower left", fontsize=20)

    fig.suptitle("")
    fig.tight_layout()
    return fig


def plot_histogram_of_timesteps(obs, stars, eta, dt_min, dt_max, Tsnap=None):
    a = (stars.ax**2 + stars.ay**2 + stars.az**2).sqrt()
    timestep = (2*eta*stars.epsilon / a).sqrt().value_in(units.Myr)

    fig, ax = pyplot.subplots()
    counts, edges = numpy.histogram(numpy.log10(timestep), bins=int(numpy.sqrt(len(stars))))
    ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid", c="k")
    ZERO = 1e-9
    if dt_min > ZERO:
        ax.axvline(numpy.log10(dt_min), c="k", ls=":", lw=1)
    else:
        ax.axvline(numpy.log10(ZERO), c="k", ls=":", lw=1)
    ax.axvline(numpy.log10(dt_max), c="k", ls=":", lw=1)
    ax.set_xlim(0.1*numpy.log10(timestep).min(), 3*numpy.log10(timestep).max())
    ax.set_xlabel("log10( Timestep [Myr] )")
    ax.set_ylabel("Count")

    if type(Tsnap) == float or type(Tsnap) == numpy.float64:
        suptitle = "{} at T={:<10.2f} Myr".format(obs.gc_name, Tsnap)
    elif Tsnap == "ICs":
        suptitle = "{} ICs".format(obs.gc_name)
    else:
        suptitle = "{}".format(obs.gc_name)
    fig.suptitle(suptitle, fontsize=22)
    return fig


class MwGcSimulation(object):
    def __init__(self, logger, obs, model_name, Nstars=1000, endtime=1000.0,
            Nsnapshots=100, code="fi", number_of_workers=4, softening=1.0, seed=-1,
            isolation=True, do_something=lambda obs, sim, stars, Tsnap, i: stars):

        self.logger = logger
        self.outdir = obs.outdir
        self.gc_slug = obs.gc_slug
        self.model_name = model_name
        self.Nstars = Nstars
        self.endtime = endtime | units.Myr
        self.Nsnapshots = Nsnapshots
        self.delta_t = self.endtime / self.Nsnapshots
        self.number_of_workers = number_of_workers
        self.softening = softening | units.parsec
        self.isolation = isolation
        self.seed = seed

        self.new_particles_cluster(obs)
        self.create_cluster_code(code)
        self.setup_bridge()

        Tstart = 0 | units.Myr  # kinda useless
        sum_energy = self.bridge.kinetic_energy + self.bridge.potential_energy
        energy0 = sum_energy.as_quantity_in(units.J)
        virial_radius = self.bridge.particles.virial_radius().as_quantity_in(units.parsec)
        print("\nTstart        : {}".format(Tstart))
        print("Energy        : {}".format(energy0))
        print("Virial radius : {}".format(virial_radius))

        self.evolve_model(do_something, obs)

        Tend = self.bridge.model_time.as_quantity_in(units.Myr)
        sum_energy = self.bridge.kinetic_energy + self.bridge.potential_energy
        energy = sum_energy.as_quantity_in(units.J)
        virial_radius = self.bridge.particles.virial_radius().as_quantity_in(units.parsec)

        print("\nTend          : {}".format(Tend))
        print("Energy        : {}".format(energy))
        print("Delta E       : {}".format((energy-energy0)/energy0))
        print("Virial radius : {}".format(virial_radius))

        self.stop()

    def new_particles_cluster(self, obs):
        # Sample initial conditions for King/Wilson/Limepy MLEs from deBoer+ 2019
        if self.model_name == "king":
            self.limepy_model, limepy_sampled, self.amuse_sampled, self.converter = \
                obs.sample_deBoer2019_bestfit_king(Nstars=self.Nstars, seed=self.seed)
        elif self.model_name == "wilson":
            self.limepy_model, limepy_sampled, self.amuse_sampled, self.converter = \
                obs.sample_deBoer2019_bestfit_wilson(Nstars=self.Nstars, seed=self.seed)
        elif self.model_name == "limepy":
            self.limepy_model, limepy_sampled, self.amuse_sampled, self.converter = \
                obs.sample_deBoer2019_bestfit_limepy(Nstars=self.Nstars, seed=self.seed)

        # Verify that the sampled profile matches the observed profile (as well as
        # the requested Limepy model, of course)
        fname = "{}{}_{}_{}_{}_{}_{}_ICs.png".format(obs.outdir, obs.gc_slug,
            self.model_name, "isolation" if self.isolation else "MWPotential2014",
            self.Nstars, self.softening.value_in(units.parsec), self.seed
        )
        plot_SigmaR_vs_R(obs, self.limepy_model, self.amuse_sampled,
            model_name=self.model_name, Tsnap="ICs",
            softening=self.softening.value_in(units.parsec)
        ).savefig(fname)
        self.logger.info("\nSaved: {0}".format(fname))

    def evolve_model(self, do_something, obs):
        # First we'd like to do_something /w ICs
        do_something(obs, self, self.amuse_sampled, self.bridge.model_time, 0)
        try:
            for i, dt in enumerate(self.delta_t * numpy.arange(1, self.Nsnapshots+1)):
                i+=1
                self.bridge.evolve_model(dt)
                print("\nTime step {}/{} --> time: {:.3f} Myr".format(i,
                    self.Nsnapshots, self.bridge.model_time.value_in(units.Myr)
                ))

                # Copy stars from gravity to output or analyze the simulation
                print(self.bridge.particles.sorted_by_attribute("key")[0:5])
                self.channel_from_gravity_to_stars.copy()
                print(self.amuse_sampled.sorted_by_attribute("key")[0:5])

                # get_particle_properties(self.amuse_sampled, w="    ")
                self.amuse_sampled = do_something(obs, self,
                    self.amuse_sampled, self.bridge.model_time, i
                )

                # If you edited the stars particle set, for example to remove stars from the
                # array because they have been kicked far from the cluster, you need to
                # copy the array back to gravity:
                self.channel_from_stars_to_gravity.copy()
        except Exception as e:
            self.bridge.stop()
            raise

    def create_cluster_code(self, code):
        print(os.environ)
        self.cluster_code = getattr(self, "new_code_"+code)()
        print("cluster_code.parameters\n", self.cluster_code.parameters)
        self.cluster_code.commit_particles()

        # Setup channels between stars particle dataset and the cluster code
        # Cannot set attributes ['ax', 'ay', 'az', 'epsilon', 'radius']
        self.channel_from_stars_to_gravity = self.amuse_sampled.new_channel_to(
            self.cluster_code.particles, attributes =[
                "mass", "x", "y", "z", "vx", "vy", "vz",
            ]
        )
        self.channel_from_gravity_to_stars = self.cluster_code.particles.new_channel_to(
            self.amuse_sampled, attributes =[
                "mass", "x", "y", "z", "vx", "vy", "vz",
                "ax", "ay", "az", "epsilon", "radius"
            ]
        )


    def setup_bridge(self):
        # If we want to test w/o bridge integrator, but /w cluster_code directly
        self.bridge = self.cluster_code
        return

        # Setup gravity bridge
        self.bridge = bridge.Bridge(use_threading=False, verbose=True)
        if self.isolation:
            # As a stability check, we first simulate the star cluster in isolation
            self.bridge.add_system(self.cluster_code, )
        else:
            # Convert galpy MWPotential2014 to AMUSE representation (Webb+ 2020)
            potential = to_amuse(MWPotential2014)
            # Stars in gravity depend on gravity from external potential MWPotential2014
            self.bridge.add_system(self.cluster_code, (potential, ))
            # External potential potential still needs to be added to system so it evolves with time
            self.bridge.add_system(potential, )

        # Set how often to update external potential
        self.bridge.timestep = self.delta_t

    def stop(self):
        self.bridge.stop()

    def new_code_fi(self):
        from amuse.community.fi.interface import Fi
        os.environ["OMP_NUM_THREADS"] = "{}".format(self.number_of_workers)
        result = Fi(self.converter, number_of_workers=1, mode="openmp",
            redirection="none")  # or "null" to silence
        result.parameters.self_gravity_flag = True
        result.parameters.use_hydro_flag = False
        result.parameters.radiation_flag = False
        # result.parameters.periodic_box_size = 500 | units.parsec
        result.parameters.timestep = 0.125 | units.Myr
        result.particles.add_particles(self.amuse_sampled)
        return result

    def new_code_ph4(self):
        from amuse.community.ph4.interface import ph4
        sys.exit(-1)
        result = ph4(mode="gpu")
        result.parameters.epsilon_squared = self.softening**2
        result.particles.add_particles(self.amuse_sampled)
        return result

    def new_code_huayno(self):
        from amuse.community.huayno.interface import Huayno
        sys.exit(-1)
        result = Huayno()
        result.parameters.epsilon_squared = self.softening**2
        result.particles.add_particles(self.amuse_sampled)
        return result

    def new_code_bhtree(self):
        from amuse.community.bhtree.interface import BHTree
        sys.exit(-1)
        result = BHTree()
        result.parameters.epsilon_squared = self.softening**2
        result.parameters.timestep = 0.125 | units.Myr
        result.particles.add_particles(self.amuse_sampled)
        return result

    def new_code_gadget2(self):
        print("new_code_gadget2")
        from amuse.community.gadget2.interface import Gadget2
        result = Gadget2(
            unit_converter=self.converter,
            number_of_workers=self.number_of_workers,
            redirection="none"
            debugger="strace",
        )
        print("new_code_gadget2 --> result.state =", result.get_name_of_current_state())

        # Gadget internal units
        code_length_unit = result.parameters.code_length_unit.as_quantity_in(units.parsec)
        code_mass_unit = result.parameters.code_mass_unit.as_quantity_in(units.MSun)
        code_time_unit = result.parameters.code_time_unit.as_quantity_in(units.Myr)
        code_velocity_unit = result.parameters.code_velocity_unit.as_quantity_in(units.km/units.s)
        print("  code_length_unit: {}".format(code_length_unit))
        print("  code_mass_unit: {}".format(code_mass_unit))
        print("  code_time_unit: {}".format(code_time_unit))
        print("  code_velocity_unit: {}".format(code_velocity_unit))

        # Get Gadget internal timesteps
        # Gadget calculate timestep for individual particles as
        # sqrt ( 2 eta epsilon / |a| ), where eta=ErrTolIntAccuracy=0.025
        # (controlled /w result.parameters.timestep_accuracy_parameter in AMUSE),
        # epsilon is the gravitational softening length, and |a| is
        # abs of acceleration (vector). GADGET-1 had 5 timestep options, but
        # GADGET2 reduced this to 1 b/c reasons, see Powers+ 2003, MNRAS, 338, 14
        self.dt_min = result.parameters.min_size_timestep.value_in(units.Myr)
        self.dt_max = result.parameters.max_size_timestep.value_in(units.Myr)
        self.eta = result.parameters.timestep_accuracy_parameter
        print("\n  min_size_timestep: {} Myr\n  max_size_timestep: {} Myr".format(
            self.dt_min, self.dt_max))
        print("  timestep_accuracy_parameter: {}".format(self.eta))

        result.parameters.time_max = 2*self.endtime
        # TODO CPU max time
        # result.parameters.time_max = 2*self.endtime

        # Set softening
        result.parameters.epsilon_squared = self.softening**2

        # Set the gadget_output_directory
        outdir = "{}{}_{}_{}_{}_{}_{}".format(self.outdir, self.gc_slug,
            self.model_name, "isolation" if self.isolation else "MWPotential2014",
            self.Nstars, self.softening.value_in(units.parsec), self.seed
        )
        if not os.path.exists(outdir) or not os.path.isdir(outdir):
            print("Created,", outdir)
            os.mkdir(outdir)
        result.parameters.gadget_output_directory = outdir

        result.particles.add_particles(self.amuse_sampled)
        return result

    def new_code_mercury(self):
        from amuse.community.mercury.interface import Mercury
        sys.exit(-1)
        result = Mercury()
        result.parameters.epsilon_squared = self.softening**2
        # result.parameters.timestep = 0.125 * self.interaction_timestep
        result.particles.add_particles(self.amuse_sampled)
        return result

    def new_code_hermite(self):
        from amuse.community.hermite.interface import Hermite
        sys.exit(-1)
        result = Hermite()
        result.parameters.epsilon_squared = self.softening**2
        result.particles.add_particles(self.amuse_sampled)
        return result

    def new_code_phigrape(self):
        from amuse.community.phigrape.interface import PhiGRAPE
        sys.exit(-1)
        result = PhiGRAPE(mode="gpu")
        result.parameters.initialize_gpu_once = 1
        result.parameters.epsilon_squared = self.softening**2
        result.particles.add_particles(self.amuse_sampled)
        return result


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
    os.environ["OMP_NUM_THREADS"] = "16"
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
