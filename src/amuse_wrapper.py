import time
import copy
import numpy
import scipy
import matplotlib
from matplotlib import pyplot

from amuse.units import units
from amuse.support.exceptions import CodeException
from mpi4py.MPI import Exception as MPIException


def print_particleset_info(p, converter, modelname):
    com = p.center_of_mass()
    Mtot = p.total_mass().as_quantity_in(units.MSun)
    Ekin = p.kinetic_energy()
    Epot = p.potential_energy()
    # try:
    #     pos, r0, rho0 = p.densitycentre_coreradius_coredens(unit_converter=converter)
    #     has_posr0rho0 = True
    # except (CodeException, MPIException) as e:
    #     has_posr0rho0 = False
    #     if "The worker application does not exist" in str(e):
    #         print("WARNING: could not start HOP worker")
    has_posr0rho0 = False

    # It seems that the total mass in the Plummer and King models is set
    # by the mass set to the unit converter. So compute Mtotal from converter.
    for i_mass, cu in enumerate(converter.units):
    # converter.units in [length, mass, time], but order could differ
        if str(cu[0]) == "mass": break
    Mtotal = (converter.units[i_mass][1]).value_in(units.MSun)

    print("Sampled {0} stars in a {1}".format(len(p), modelname))
    print("  Mtotal (requested): {0}, Mtot (sampled): {1}".format(Mtotal, Mtot))
    print("  Ekin (sampled): {0:.2e} J ({1:.2e} erg) [{2:.2e} Msun/kms**2]".format(
        Ekin.value_in(units.J), Ekin.value_in(units.erg),
        Ekin.value_in(units.MSun*units.kms**2) ))
    print("  Epot (sampled): {0:.2e} J ({1:.2e} erg) [{2:.2e} Msun/kms**2]".format(
        Epot.value_in(units.J), Epot.value_in(units.erg),
        Epot.value_in(units.MSun*units.kms**2) ))
    print("  Virial ratio (-2*Ekin / Epot): {0}".format(-2*Ekin/Epot))
    print("  CoM (sampled): {0}".format(com.as_quantity_in(units.pc)))
    if has_posr0rho0:
        print("  \nOutput of densitycentre_coreradius_coredens")
        print("    pos: {0}".format(pos))
        print("    r0: {0:.2e} parsec,  rho0: {1:.2e} MSun/parsec**3".format(
            r0.value_in(units.parsec), rho0.value_in(units.MSun/units.parsec**3)))
    print("")


def get_radial_profiles(p, c=None, rmin=1e-3, rmax=1e3, Nbins=256, timing=True):
    """ Generate radial profile of Particleset p /w center c """

    print("\nget_radial_profiles")
    print("  com:  {}".format(p.center_of_mass().value_in(units.parsec)))
    print("  comvel:  {}".format(p.center_of_mass_velocity().value_in(units.km/units.s)))
    Mtot = p.total_mass().as_quantity_in(units.MSun)
    print("  Mtot:   {}".format(Mtot))
    Ekin = p.kinetic_energy().as_quantity_in(units.J)
    print("  Ekin:   {}".format(Ekin))
    Epot = p.potential_energy().as_quantity_in(units.J)
    print("  Epot:   {}".format(Epot))
    print("  Ekin/Epot: {}".format(Ekin/Epot))
    Ltot = p.total_angular_momentum().as_quantity_in(units.MSun*units.parsec**2/units.Myr)
    print("  Ltot:   {}".format(Ltot))
    ptot = p.total_momentum().as_quantity_in(units.MSun*units.parsec/units.Myr)
    print("  ptot:      {}".format(ptot))
    print("")

    start = time.time()

    if c is None:
        c = [0.0, 0.0, 0.0]

    r = numpy.sqrt(
        (p.x.value_in(units.parsec) - c[0])**2 +
        (p.y.value_in(units.parsec) - c[1])**2 +
        (p.z.value_in(units.parsec) - c[2])**2)
    r_i = r.argsort()

    r_edges = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), Nbins+1)
    r_mid = 0.5 * (r_edges[1:] + r_edges[:-1])
    dr = r_edges[1:] - r_edges[:-1]
    volume = 4 * numpy.pi * (r_mid**2) * dr
    N_in_shell = numpy.array( [numpy.abs(r[r_i]-rr).argmin() for rr in r_edges] )

    M_of_r = numpy.zeros(Nbins)
    rho_of_r = numpy.zeros(Nbins)
    mass_sorted = p.mass.value_in(units.MSun)[r_i]
    for i in range(1, len(r_edges)-1):
        M_of_r[i] = ( mass_sorted[N_in_shell[i-1]:N_in_shell[i]].sum() )
        rho_of_r[i] = M_of_r[i]
    rho_of_r = (rho_of_r / volume) | units.MSun/units.parsec**3

    if timing:
        print("get_radial_profiles took {0:.2f} s".format(time.time() - start))

    return r_mid | units.parsec, N_in_shell, M_of_r.cumsum() | units.MSun, \
        rho_of_r, volume | units.parsec**-3


def project_amuse_profiles(radii, rho_of_r, timing=True):
    # Shamelessly copied from mgieles/limepy/limepy.py, but see
    # 2015MNRAS.454..576G eq. 35
    start = time.time()
    R = copy.copy(radii.value_in(units.parsec))
    radii = radii.value_in(units.parsec)
    rho = rho_of_r.value_in(units.MSun/units.parsec**3)
    Sigma = numpy.zeros(len(radii))
    for i in range(len(radii)-1):
        c = (radii >= R[i])
        r = radii[c]
        z = numpy.sqrt(abs(r**2 - R[i]**2)) # avoid small neg. values
        Sigma[i] = 2.0*abs(scipy.integrate.simps(rho[c], x=z))
    if timing:
        print("_project_amuse took {0:.2f} s".format(time.time() - start))

    # TODO: Velocity dispersion profiles
    return R, Sigma  # v2_PM, v2_RV



def plot_radial_profiles(radii, N_in_shell, M_below_r, rho_of_r, volume,
        rmin=1e-2, rmax=1e3, fig=None, has_tex=True):

    if not has_tex:
        matplotlib.rcParams.update({"text.usetex": False})

    matplotlib.rcParams.update({"font.size": 16})
    if fig is None:
        fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2, figsize=(16, 16))
    else:
        ax1, ax2, ax3, ax4 = fig.axes

    # Sampled number of stars
    ax1.plot(radii.value_in(units.parsec), N_in_shell,
        c="r", lw=2, drawstyle="steps-mid", label="sampled")
    # ax1.plot(radii.value_in(units.parsec), phi_of_r,
    #     c="r", lw=2, drawstyle="steps-mid", label="sampled")
    ax1.set_xscale("log")
    ax1.set_xlabel("Radius [parsec]")
    ax1.set_ylabel("Count")

    # Sampled mass
    ax2.plot(radii.value_in(units.parsec), M_below_r.value_in(units.MSun),
        c="r", lw=2, drawstyle="steps-mid", label="sampled")
    ax2.set_xscale("log")
    ax2.set_xlabel("Radius [parsec]")
    ax2.set_ylabel("Mass (< r) [MSun]")

    # Sampled density
    ax3.plot(radii.value_in(units.parsec),
        rho_of_r.value_in(units.MSun/units.parsec**3),
        c="r", lw=2, drawstyle="steps-mid", label="sampled")
    ax3.set_ylim(
        0.1*numpy.mean(rho_of_r[-16:].value_in(units.MSun/units.parsec**3)),
        1.2*rho_of_r[0].value_in(units.MSun/units.parsec**3)
    )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Radius [parsec]")
    ax3.set_ylabel("Density [MSun / parsec**3]")

    ax4.plot(radii.value_in(units.parsec),
        N_in_shell/volume, c="r", lw=2,
        drawstyle="steps-mid", label="sampled")
    # instarmin = numpy.where(radii.value_in(units.parsec) > rmax)[0][0]
    # ax4.set_ylim((N_in_shell/volume)[instarmin], 3*numpy.mean((N_in_shell/volume)[0:10]))
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Radius [parsec]")
    ax4.set_ylabel("Number Density [1 / parsec**3]")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(rmin, rmax)
        ax.legend()

    return fig, (ax1, ax2, ax3, ax4)


def scatter_particles_xyz(p, draw_max=1337, unit_length=units.parsec,
        show_bound=True, labels=[], plot_r=False):
    # p is an instance of amuse.datamodel.particles.Particles

    # Create the figure
    from matplotlib import gridspec
    fig = pyplot.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(3, 3)
    axxz = fig.add_subplot(gs.new_subplotspec((0, 0), colspan=2))
    axzy = fig.add_subplot(gs.new_subplotspec((1, 2), rowspan=2))
    axxy = fig.add_subplot(gs.new_subplotspec((1, 0), colspan=2, rowspan=2))
                          #sharex=axxz, sharey=axzy)
    axt = fig.add_subplot(gs.new_subplotspec((0, 2)))
    axt.axis("off")
    # gs.update(wspace=0, hspace=0, top=0.94, left=0.15)

    # Select a subset of all particles
    from numpy.random import choice
    draw_max = len(p) if len(p) < draw_max else draw_max
    mask = choice(len(p), int(draw_max), replace=False)
    print("Plotting draw_max = {0} particles".format(draw_max))

    # TODO: select on bound/unbound
    if show_bound:
        center_of_mass = p.center_of_mass()
        bound = p.select(
            lambda r: (center_of_mass - r).length()  < 1.0 | units.parsec,
            ["position"]
        )
        outersphere = p.select(
            lambda r: (center_of_mass - r).length() >= 1.0 | units.parsec,
            ["position"]
        )

    # Set the limits
    xlim = ylim = 1.5*p.total_radius().value_in(unit_length)

    # 'Main' panel: XY-plane
    axxy.plot(p.x[mask].value_in(unit_length), p.y[mask].value_in(unit_length),
        "rD", ms=0.01 if draw_max > 10000 else 1, rasterized=True)
    if show_bound:
        axxy.plot(
            bound.x.value_in(unit_length),
            bound.y.value_in(unit_length),
            "gD", ms=0.1, alpha=0.7, rasterized=True
        )
    axxy.set_xlim([-xlim, xlim])
    axxy.set_ylim([-ylim, ylim])
    axxy.set_xlabel("x [{0}]".format(unit_length))
    axxy.set_ylabel("y [{0}]".format(unit_length))
    # axxy.set_aspect(1)

    # Top panel: XZ
    axxz.plot(p.x[mask].value_in(unit_length), p.z[mask].value_in(unit_length),
        "rD", ms=0.01 if draw_max > 10000 else 1, rasterized=True)
    if show_bound:
        axxz.plot(
            bound.x.value_in(unit_length),
            bound.z.value_in(unit_length),
            "gD", ms=0.1, alpha=0.7, rasterized=True
        )
    axxz.set_xlim([-xlim, xlim])
    axxz.set_ylim([-ylim/2, ylim/2])

    axxz.set_ylabel("z [{0}]".format(unit_length))
    # axxz.set_aspect(1)

    # Right panel: ZY
    axzy.plot(p.z[mask].value_in(unit_length), p.y[mask].value_in(unit_length),
        "rD", ms=0.01 if draw_max > 10000 else 1, rasterized=True)
    if show_bound:
        axzy.plot(
            bound.z.value_in(unit_length),
            bound.y.value_in(unit_length),
            "gD", ms=0.1, alpha=0.7, rasterized=True
        )
    axzy.set_xlim([-xlim/2, ylim/2])
    axzy.set_ylim([-ylim, ylim])
    axzy.set_xlabel("z [{0}]".format(unit_length))
    # axzy.set_aspect(1)

    for i, label in enumerate(labels):
        axt.text(0.5, 0.9-0.1*i, label, fontsize=16,
            ha="center", va="center", transform=axt.transAxes)
    fig.subplots_adjust(wspace=0, hspace=0, left=0.09, right=0.98, bottom=0.07, top=0.98)

    if plot_r or True:
        r = p.total_radius().value_in(unit_length)  # is max((x**2 + y**2 + z**2).sqrt())
        print("Total radius: {0:.2f} {1}".format(r, unit_length))
        rvir = p.virial_radius().value_in(unit_length)
        print("Virial radius: {0:.2f} {1}".format(rvir, unit_length))

        phi = numpy.arange(0, 2*numpy.pi, 0.01)
        axxz.plot(r * numpy.cos(phi), r * numpy.sin(phi), c="k", lw=2, ls="-")
        axzy.plot(r * numpy.cos(phi), r * numpy.sin(phi), c="k", lw=2, ls="-")
        axxy.plot(r * numpy.cos(phi), r * numpy.sin(phi), c="k", lw=2, ls="-")

    # axxz, axzy, axxy, axt = fig.axes
    return fig


if __name__ == "__main__":
    Nstars = 50000
    Mtotal = 30000 | units.MSun
    Rtidal = 1 | units.parsec
    rmin = 1e-2  # parsec
    rmax = 1e3   # parsec

    from amuse.units import nbody_system
    convert_nbody = nbody_system.nbody_to_si(Mtotal, Rtidal)
    print("converter.length = {0} parsec\nconverter.mass = {1} MSun\n"
          "converter.time = {2} yr".format(
        (convert_nbody.units[0][1]).value_in(units.parsec),
        (convert_nbody.units[1][1]).value_in(units.MSun),
        (convert_nbody.units[2][1]).value_in(units.yr)
    ))

    from amuse.ic.plummer import new_plummer_sphere
    plummer = new_plummer_sphere(Nstars, convert_nbody)

    print_particleset_info(plummer, convert_nbody, "Plummer Sphere")
    a = 3*numpy.pi/16 | units.parsec  # b/c AMUSE default

    radii, N_in_shell, M_below_r, rho_of_r, volume = \
        get_radial_profiles(plummer, rmin=rmin, rmax=rmax, Nbins=256)

    from plummer import add_plummer_radii_to_ax
    from plummer import add_plummer_mass_profile_to_ax
    from plummer import add_plummer_density_profile_to_ax

    r_ana = numpy.logspace(-3, 4, 64) | units.parsec
    fig, (ax1, ax2, ax3) = plot_radial_profiles(
        plummer, radii, N_in_shell, M_below_r, rho_of_r, volume, rmin, rmax
    )
    add_plummer_mass_profile_to_ax(ax2, plummer, a, r_ana)
    add_plummer_density_profile_to_ax(ax3, plummer, a, r_ana)
    for ax in [ax1, ax2, ax3]:
        add_plummer_radii_to_ax(ax, plummer, a)
        ax.legend()
    fig.tight_layout()
    fig.show()

    fig = scatter_particles_xyz(plummer, labels=["Plummer Sphere"])
    fig.show()
