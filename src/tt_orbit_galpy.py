import numpy
import scipy
import scipy.linalg
from scipy import signal
from astropy import units
from matplotlib import pyplot
pyplot.style.use("tlrh")

from galpy.orbit import Orbit
from galpy.util import bovy_plot
from galpy.util import bovy_coords
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


def plot_orbit_xyz(o, t=None, fig=None):
    if not fig:
        fig = pyplot.figure(figsize=(6, 10))
        ax1 = pyplot.axes([0.00, 0.40, 1.00, 0.60])
        ax2 = pyplot.axes([0.00, 0.10, 1.00, 0.30])
    else:
        fig, (ax1, ax2) = fig, fig.axes

    if hasattr(o, "name"):
        s = o.name
        if type(o.time()) == numpy.ndarray:
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
        if type(o.time()) == numpy.ndarray:
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
    # Calculate ttensor and the eigenvalues
    tt = ttensor(MWPotential2014, o.R(o.time()), o.z(o.time()), t=o.time())
    eigenvals = numpy.array([scipy.linalg.eigvals(ttij) for ttij in tt.T], dtype="float128")
    norm_eigenvals = numpy.linalg.norm(eigenvals, axis=1)

    # Find peaks in Tij
    peaks, properties = scipy.signal.find_peaks(norm_eigenvals)  # , threshold=1e-3
    print("\npeaks:\n", peaks)
    print("properties:\n", properties)
    print("time:\n", o.time()[peaks])
    print("")

    # And plot it
    if not fig_axes:
        fig, (ax1, ax2, ax3, ax4) = pyplot.subplots(4, 1, figsize=(18, 12))
    else:
        fig, ax1, ax2, ax3, ax4 = fig_axes

    if hasattr(o, "name"):
        s = o.name
        if type(o.time()) == numpy.ndarray:
            s += r" (t $\in$ {0:.2f} - {1:.2f} Gyr)".format(o.time()[0], o.time()[-1])
        ax1.set_title(s, fontsize=22)

    # What are the units? Are these calculations correct? IHaveNoIdeaWhatIamDoing
    ax1.plot(o.time(), norm_eigenvals, label=r"$|T_{ij}|$")
    ax1.set_ylabel(r"$|T_{ij}|$")
    ax1.set_xticklabels([], [])
    ax1.legend(frameon=False, fontsize=18)

    ax2.plot(o.time(), o.R(o.time()), label="2D")
    ax2.plot(o.time(), o.r(o.time()), label="3D")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel(r"Radius [kpc]")
    ax2.set_xticklabels([], [])
    ax2.legend(frameon=False, fontsize=18)

    # What's causing the spikes?
    ax3.plot(o.time(), eigenvals[::,0], c="r", label=r"$T_{ij}$[0]")
    ax3.plot(o.time(), eigenvals[::,1], c="g", label=r"$T_{ij}[1]$")
    ax3.plot(o.time(), eigenvals[::,2], c="b", label=r"$T_{ij}[2]$")
    ax3.set_ylabel(r"$T_{ij}$")
    ax3.set_xticklabels([], [])
    ax3.legend(frameon=False, fontsize=18)

    ax4.plot(o.time(), o.x(o.time()), c="r", label="x")
    ax4.plot(o.time(), o.y(o.time()), c="g", label="y")
    ax4.plot(o.time(), o.z(o.time()), c="b", label="z")
    ax4.axhline(0, c="k", ls="--")
    ax4.set_xlabel("Time [Gyr]")
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel(r"Position [kpc]")
    ax4.legend(frameon=False, fontsize=18)

    for ax in fig.axes:
        for peak_time in o.time()[peaks]:
            ax.axvline(peak_time, c="k", ls=":")

    # fig.tight_layout()
    fig.subplots_adjust(hspace=0)

    return fig, peaks


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
