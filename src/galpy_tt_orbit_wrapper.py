import warnings
import numpy
import scipy
import scipy.linalg
from scipy import signal
from astropy import units
import matplotlib
from matplotlib import pyplot
pyplot.style.use("tlrh")

from galpy.orbit import Orbit
from galpy.util import bovy_plot
from galpy.util import bovy_coords
from galpy.util import bovy_conversion
from galpy.potential import ttensor
from galpy.potential import MWPotential2014
from galpy.potential import evaluatePotentials


def integrate_and_print_info(o, ts=numpy.linspace(0.0, 100.0, 8096+1), verbose=True):
    if verbose: print( "Started at time {0:.2f} Gyr ..".format(o.time() ))  # time -> float
    o.integrate(ts, MWPotential2014)
    if verbose: print( "  .. integrated until {0:.2f} Gyr\n".format(o.time()[-1] ))  # time -> array

    # print("Pericenter: {0:.2f} kpc".format(o.rperi()))
    # print("Apocenter: {0:.2f} kpc".format(o.rap()))
    # print("Eccentricity: {0:.2f}".format(o.e()))
    # print("Max. heigth above plane: {0:.2f} kpc".format(o.zmax()))

    # print("Energy of the orbit: {0:.2f} (km/s)^2".format(o.E()))


    print("  ro: {0:.1f} kpc\n  vo: {1:.1f} km/s\n  zo: {2:.1f} pc\n  solarmotion: {3} km/s\n".format(
        o._ro, o._vo, 1000*o._zo, o._solarmotion))  # sanity check
    print("  RA    = {0: 7.2f} deg".format(o.ra()))
    print("  Dec   = {0: 7.2f} deg".format(o.dec()))
    print("  dist  = {0: 7.2f} kpc".format(o.dist()))
    print("  pmra  = {0: 7.2f} mas/yr".format(o.pmra()))
    print("  pmdec = {0: 7.2f} mas/yr".format(o.pmdec()))
    print("  vlos  = {0: 7.2f} km/s".format(o.vlos()))

    # Galpy X=0 sits at Sun, points towards Galactic center?
    # For Baumgardt+ 2019 X=0 sits at Galactic center, points towards Sun?
    print("  X     = {0: 7.2f} kpc".format(o.helioX()))  # o._ro - o.helioX() to be consistent with B19
    print("  Y     = {0: 7.2f} kpc".format(o.helioY()))
    print("  Z     = {0: 7.2f} kpc".format(o.helioZ()))
    print("  U     = {0: 7.2f} km/s".format(o.U()))
    print("  V     = {0: 7.2f} km/s".format(o.V()))
    print("  W     = {0: 7.2f} km/s".format(o.W()))
    print("  Rper  = {0: 7.2f} kpc".format(o.rperi()))
    print("  Rapo  = {0: 7.2f} kpc".format(o.rap()))


def plot_orbit_xyz(o, t=None, fig=None):
    if not fig:
        fig = pyplot.figure(figsize=(6, 10))
        ax1 = pyplot.axes([0.00, 0.40, 1.00, 0.60])
        ax2 = pyplot.axes([0.00, 0.10, 1.00, 0.30])
    else:
        fig, (ax1, ax2) = fig, fig.axes

    if hasattr(o, "name"):
        s = o.name
        if type(o.time()) == numpy.ndarray:  # b/c o.time() is in Gyr
            s += " (t $\in$ {0:.2f} - {1:.2f} Gyr)".format(o.time()[0], o.time()[-1])
        ax1.set_title(s, fontsize=16)

    pyplot.sca(ax1); o.plot(d1="x", d2="y", gcf=True)
    if t is not None:
        ax1.plot([o.x(t)], [o.y(t)], "ro", ms=8)
    ax1.yaxis.set_label_position("right")

    pyplot.sca(ax2); o.plot(d1="x", d2="z", gcf=True)
    if t is not None:
        ax2.plot([o.x(t)], [o.z(t)], "ro", ms=8)
    ax2.yaxis.tick_right()

    return fig


def plot_orbit_Rz(o, t=None, fig=None):
    # [("R", "z"), ("x","y"), ("x","z"), ("R","vR"), ("ra","dec"), ("ll","bb")]
    if not fig:
        fig = pyplot.figure(figsize=(6, 10))
        ax1 = pyplot.axes([0.00, 0.50, 1.00, 0.50])
        ax2 = pyplot.axes([0.00, 0.00, 1.00, 0.50])
    else:
        fig, (ax1, ax2) = fig, fig.axes

    if hasattr(o, "name"):
        s = o.name
        if type(o.time()) == numpy.ndarray:
            s += " (t $\in$ {0:.2f} - {1:.2f} Gyr)".format(o.time()[0], o.time()[-1])
        ax1.set_title(s, fontsize=16)

    pyplot.sca(ax1); o.plot(d1="R", d2="z", gcf=True)
    if t is not None:
        ax1.plot([o.R(t)], [o.z(t)], "ro", ms=8)
    ax1.yaxis.set_label_position("right")

    pyplot.sca(ax2); o.plot(d1="R", d2="vR", gcf=True)
    if t is not None:
        ax2.plot([o.R(t)], [o.vR(t)], "ro", ms=8)
    ax2.yaxis.tick_right()

    return fig


def plot_potential(o, t=None, fig=None):
    if not fig:
        fig = pyplot.figure(figsize=(6, 10))
        ax1 = pyplot.axes([0.00, 0.40, 1.00, 0.60])
        ax2 = pyplot.axes([0.00, 0.10, 1.00, 0.30])
    else:
        fig, (ax1, ax2) = fig, fig.axes

    if hasattr(o, "name"):
        s = o.name
        if type(o.time()) == numpy.ndarray:  # b/c o.time() is in Gyr
            s += r" (t $\in$ {0:.2f} - {1:.2f} Gyr)".format(o.time()[0], o.time()[-1])
        # ax1.text(s, 0.5, 1.01, ha="center", va="bottom", transform=ax1.transAxes)
        ax1.set_title(s, fontsize=16)

    # because plotPotentials deploys use_physical=False in evaluatePotentials
    o.turn_physical_off()

    pyplot.sca(ax1)
    plotPotentialsHack(MWPotential2014, rmin=-2.0, rmax=2.0, nrs=202,
        zmin=-2.0, zmax=2.0, nzs=202, xy=True, gcf=True)
    if t is not None:
        ax1.plot([o.x(t)], [o.y(t)], "ro", ms=8)
    ax1.yaxis.set_label_position("right")
    ax1.set_xticklabels([], [])
    ax1.set_aspect(1)

    pyplot.sca(ax2)
    plotPotentialsHack(MWPotential2014, rmin=-2.0, rmax=2.0, nrs=202,
        zmin=-1.0, zmax=1.0, nzs=202, xz=True, gcf=True)
    if t is not None:
        ax2.plot([o.x(t)], [o.z(t)], "ro", ms=8)
    ax2.yaxis.tick_right()
    ax2.set_aspect(1)

    # Restore Balance to the Uni(ts)verse ;-)
    o.turn_physical_on()

    return fig


# Added *args, **kwargs function arguments, and added xz support
def plotPotentialsHack(
        Pot, rmin=0., rmax=1.5, nrs=21, zmin=-0.5, zmax=0.5, nzs=21,
        phi=None, xy=False, xz=False, t=0., effective=False, Lz=None, ncontours=21,
        aspect=None,justcontours=False, levels=None, cntrcolors="k",
        *args, **kwargs,
    ):

    Rs = numpy.linspace(rmin, rmax, nrs)
    zs = numpy.linspace(zmin, zmax, nzs)
    potRz = numpy.zeros((nrs, nzs))
    for ii in range(nrs):
        for jj in range(nzs):
            if xy:
                R, phi, z = bovy_coords.rect_to_cyl(Rs[ii], zs[jj], 0.)
            elif xz:  # Hack b/c I want this
                R, phi, z = bovy_coords.rect_to_cyl(Rs[ii], 0., zs[jj])
            else:
                R, z = Rs[ii], zs[jj]
            potRz[ii, jj] = evaluatePotentials(
                Pot, numpy.fabs(R), z, phi=phi, t=t,use_physical=False
            )
        if effective:
            potRz[ii, :] += 0.5*Lz**2/Rs[ii]**2.

    if aspect is None:
        aspect = .75*(rmax-rmin) / (zmax-zmin)
    if xy:
        xlabel = r"$x/R_0$"
        ylabel = r"$y/R_0$"
    elif xz:
        xlabel = r"$x/R_0$"
        ylabel = r"$z/R_0$"
    else:
        xlabel = r"$R/R_0$"
        ylabel = r"$z/R_0$"
    if levels is None:
        levels = numpy.linspace(numpy.nanmin(potRz), numpy.nanmax(potRz), ncontours)

    return bovy_plot.bovy_dens2d(
        potRz.T, origin="lower", cmap="gist_gray", contours=True,
        xlabel=xlabel, ylabel=ylabel, aspect=aspect,
        xrange=[rmin, rmax], yrange=[zmin, zmax],
        cntrls="-", justcontours=justcontours,
        levels=levels, cntrcolors=cntrcolors, *args, **kwargs)


def plot_ttensor(o, fig_axes=None):
    # Calculate ttensor and the eigenvalues in Galpy's natural units
    tt = ttensor(MWPotential2014, o.R(o.t), o.z(o.t), t=o.t)
    # ComplexWarning: Casting complex values to real discards the imaginary part
    with warnings.catch_warnings(record=True) as w:
        eigenvals = numpy.array([scipy.linalg.eigvals(ttij) for ttij in tt.T], dtype="float128")
    # norm_eigenvals is sqrt(p1^2+p2^2+p3^2) where p1,p2,p3 eigenvalues of Tij
    norm_eigenvals = numpy.linalg.norm(eigenvals, axis=1)

    # Find peaks in Tij
    # peaks are array indices, o.time()[peaks] gives time in Gyr,
    # and o.t[peaks] gives time in Galpy natural units
    peaks, properties = scipy.signal.find_peaks(norm_eigenvals)  # , threshold=1e-3
    # Time in results_half are 'unit bin'
    results_half = scipy.signal.peak_widths(norm_eigenvals, peaks, rel_height=0.5)
    # ttensor in Galpy's natural units --> peak properties also in natural units
    time_range = o.time()[-1] - o.time()[0]
    bin_to_Gyr = time_range / len(o.time())
    Gyr_to_Myr = 1000.

    print("\npeaks:\n", peaks)
    print("properties:\n", properties)
    print("time:\n", o.time()[peaks])
    print("peak_widths [Myr]:\n", results_half[0]*bin_to_Gyr*Gyr_to_Myr)
    print("peak_heights:\n", results_half[1])
    print("peak_left_ips [Gyr]:\n", results_half[2]*bin_to_Gyr)
    print("peak_right_ips [Gyr]:\n", results_half[3]*bin_to_Gyr)
    print("")

    # And plot it
    if not fig_axes:
        fig, (ax1, ax2, ax3, ax4) = pyplot.subplots(4, 1, figsize=(18, 12))
    else:
        fig, ax1, ax2, ax3, ax4 = fig_axes

    if hasattr(o, "name"):
        s = o.name
        if type(o.time()) == numpy.ndarray:  # b/c o.time() is in Gyr
            s += r" (t $\in$ {0:.2f} - {1:.2f} Gyr)".format(o.time()[0], o.time()[-1])
        ax1.set_title(s, fontsize=22)

    # Unit time [Gyr] for x-axis, unit ?? for y-axis
    ax1.plot(o.time(), norm_eigenvals)
    ax1.set_ylabel(r"sqrt($p_1^2+p_2^2+p_3^2$)", fontsize=12)
    ax1.set_xticklabels([], [])
    ax1.legend(frameon=False, fontsize=18)

    ax2.plot(o.time(), o.R(o.t), label="R")
    ax2.plot(o.time(), o.r(o.t), label="r")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel(r"Radius [kpc]")
    ax2.set_xticklabels([], [])
    ax2.legend(frameon=False, fontsize=18)

    # What's causing the spikes?
    ax3.plot(o.time(), eigenvals[::,0], c="r", label=r"$p_1$")
    ax3.plot(o.time(), eigenvals[::,1], c="g", label=r"$p_2$")
    ax3.plot(o.time(), eigenvals[::,2], c="b", label=r"$p_3$")
    ax3.set_ylabel(r"$T_{ij}$")
    ax3.set_xticklabels([], [])
    ax3.legend(frameon=False, fontsize=18)

    ax4.plot(o.time(), o.x(o.t), c="r", label="x")
    ax4.plot(o.time(), o.y(o.t), c="g", label="y")
    ax4.plot(o.time(), o.z(o.t), c="b", label="z")
    ax4.axhline(0, c="k", ls="--")
    ax4.set_xlabel("Time [Gyr]")
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel(r"Position [kpc]")
    ax4.legend(frameon=False, fontsize=18)

    for i, ax in enumerate(fig.axes):
        for peak_time in o.time()[peaks]:
            ax.axvline(peak_time, c="k", ls=":")
            if i is 0:
                trans = matplotlib.transforms.blended_transform_factory(
                    ax.transData, ax.transAxes)
                ax.text(peak_time, 1.0, "{:.2f}".format(peak_time),
                    c="k", fontsize=10, ha="left", va="top", rotation=45, transform=trans)
    for i, (w, y, xmin, xmax) in enumerate(zip(
            results_half[0]*bin_to_Gyr*Gyr_to_Myr, results_half[1],
            results_half[2]*bin_to_Gyr, results_half[3]*bin_to_Gyr
        )):
        ax1.hlines(y=y, xmin=xmin, xmax=xmax, color="purple", linewidth=4)
        ax1.text(xmax, y, "{:.2f} Myr".format(w), c="k", fontsize=12,
            ha="left", va="center", rotation=45, transform=ax1.transData)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)

    return fig, peaks, results_half


if __name__ == "__main__":
    print("Hello, world!")

    plotPotentialsHack(MWPotential2014, nrs=202, nzs=202)
    ax = pyplot.gca(); ax.set_aspect(1)
    # Colorbar breaks :-(
    pyplot.show()

    plotPotentialsHack(MWPotential2014, rmin=-0.5, rmax=0.5, nrs=202,
        zmin=-0.5, zmax=0.5, nzs=202, xy=True)
    ax = pyplot.gca(); ax.set_aspect(1)
    pyplot.show()

    plotPotentialsHack(MWPotential2014, rmin=-0.5, rmax=0.5, nrs=202,
        zmin=-0.5, zmax=0.5, nzs=202, xz=True)
    ax = pyplot.gca(); ax.set_aspect(1)
    pyplot.show()

    plotPotentialsHack(MWPotential2014, rmin=-0.5, rmax=0.5, nrs=202,
        zmin=-0.5, zmax=0.5, nzs=202, xy=True, xz=True)  # should xy=True and ignore xz
    ax = pyplot.gca(); ax.set_aspect(1)
    pyplot.show()
