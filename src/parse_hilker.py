import os
import numpy
import logging
from matplotlib import pyplot
from astropy import units as u
from astropy import constants as const
from galpy.orbit import Orbit
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

LOGGER = logging.getLogger()
BASEDIR = "../data/"


def tcross(rho):
    """ rho: astropy quantity """
    return (1 / numpy.sqrt(const.G * rho)).to(u.Myr).value


def parse_hilker_2019_combined(logger, fname="{0}combined_table.txt".format(BASEDIR), debug=False):
    # https://people.smp.uq.edu.au/HolgerBaumgardt/globular/combined_table.txt
    if not os.path.isfile(fname) or not os.path.exists(fname):
        logger.error("ERROR: file not found: {0}".format(fname))
        return

    if debug:
        logger.debug("\nParsing Hilker+ (2019) combined table")

    # TODO: something iffy is going on b/c we have two more columns in each
    # row than the header indicates ...
    names = [
        "Cluster", "RA", "DEC", "R_Sun", "R_GC",
        "Mass", "DM", "V", "V_err", "M/L_V", "M/L_V_err", "rc",
        "rh,l", "rh,m", "rt", "rho_c", "rho_h,m",
        "sig_c", "sig_h,m", "lg(Trh)", "MF", "F_REM",
        "sig0", "vesc", "etac", "etah",
    ]
    dtype = [
        "U16", "float", "float", "float", "float",
        "float", "float", "float", "float", "float",
        "float", "float", "float", "float", "float",
        "float", "float", "float", "float", "float",
        "float", "float", "float", "float", "float",
        "float",
    ]
    delimiter = [
        14, 10, 11, 8, 9,
        12, 12, 6, 6, 7,
        8, 7, 8, 9, 10,
        7, 8, 8, 10, 6,
        8, 8, 6, 8, 6,
        7,
    ]

    if debug and False:
        logger.debug("\nnames:     {}\ndtype:     {}\ndelimiter: {}\n".format(
        len(names), len(dtype), len(delimiter) ))

        logger.debug("-"*45)
        logger.debug("{0:<15s}{1:<15s}{2:<15s}".format("name", "dtype", "delimiter"))
        logger.debug("-"*45)
        for i in range(len(names)):
            logger.debug("{0:<15s}{1:<15s}{2:<15d}".format(names[i], dtype[i], delimiter[i]))
        logger.debug("-"*45 + "\n")

    data = numpy.genfromtxt(fname, skip_header=2, delimiter=delimiter,
        dtype=dtype, names=names, autostrip=True)
    if debug:
        logger.debug("\nHere is the first entry:")
        for n in data.dtype.names:
            logger.debug("{0:<20s}{1}".format(n, data[0][n]))

        logger.debug("\ndelimiter.cumsum()\n{0}\n".format(numpy.array(delimiter).cumsum()))

        logger.debug("\nHere are the first five rows:")
        for i in range(5): logger.debug(data[i])

        logger.debug("\nHere are the colums Cluster"+
            "of the first five rows")
        logger.debug(data["Cluster"][0:5])

    return data


def parse_hilker_2019_velocity_dispersions(logger, fname="{0}rv.dat".format(BASEDIR), debug=False):
    # https://people.smp.uq.edu.au/HolgerBaumgardt/globular/rv.dat
    # The following Table contains the velocity dispersion profiles of 139
    # Galactic globular clusters. The Table is based on the following papers:
    #   - Watkins et al. (2015), ApJ 803, 29
    #   - Baumgardt (2017), MNRAS 464, 2174
    #   - Kamann et al. (2018), MNRAS, 473, 5591
    #   - Baumgardt & Hilker (2018), MNRAS 478, 1520
    #   - Baumgardt, Hilker, Sollima & Bellini (2019), MNRAS 482, 5138

    if not os.path.isfile(fname) or not os.path.exists(fname):
        logger.error("ERROR: file not found: {0}".format(fname))
        return

    if debug:
        logger.debug("\nParsing Hilker+ (2019) velocity dispersion profiles")

    # https://people.smp.uq.edu.au/HolgerBaumgardt/globular/veldis.html
    # does have a column NStar, but that column is not available for
    # https://people.smp.uq.edu.au/HolgerBaumgardt/globular/rv.dat
    names = [
        "Cluster", "radius", "velocity_dispersion",
        "velocity_dispersion_err_up", "velocity_dispersion_err_down",
        "type",
    ]
    dtype = [
        "S16", "float", "float", "float", "float", "S16"
    ]
    delimiter = [
        14, 7, 6, 6, 6, 6
    ]

    if debug and False:
        logger.debug("\nnames:     {}\ndtype:     {}\ndelimiter: {}\n".format(
        len(names), len(dtype), len(delimiter) ))

        logger.debug("-"*45)
        logger.debug("{0:<15s}{1:<15s}{2:<15s}".format("name", "dtype", "delimiter"))
        logger.debug("-"*45)
        for i in range(len(names)):
            logger.debug("{0:<15s}{1:<15s}{2:<15d}".format(names[i], dtype[i], delimiter[i]))
        logger.debug("-"*45 + "\n")

    data = numpy.genfromtxt(fname, skip_header=0, delimiter=delimiter,
        dtype=dtype, names=names, autostrip=True)
    if debug:
        logger.debug("\nHere is the first entry:")
        for n in data.dtype.names:
            logger.debug("{0:<40s}{1}".format(n, data[0][n]))

        logger.debug("\ndelimiter.cumsum()\n{0}\n".format(numpy.array(delimiter).cumsum()))

        logger.debug("\nHere are the first five rows:")
        for i in range(5): logger.debug(data[i])

        logger.debug("\nHere are the colums Cluster"+
            "of the first five rows")
        logger.debug(data["Cluster"][0:5])

    return data


def print_hilker_vs_simbad(clusters, H19_c, B19, w1=140, w2=12):
    print("{:^152}".format("SIMBAD vs. Hilker+ 2019"))
    print("|" + "-"*w1 + "|" + "-"*w2 + "|")
    print("|{:<12s}{:<12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>15s}{:>6}{:^12s}|".format(
        "Reference", "Name", "RA", "DEC", "R_Sun", "rad. vel.", "muRA", "muDEC", "Mass", "r_h,m",
        "log(rho_h,m)", "|", "tcross"))
    print("|{:<12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>15s}{:>6}{:^12s}|".format(
        "", "", "deg", "deg", "kpc", "km/s", "mas/yr", "mas/yr", "MSun", "pc", "MSun/pc^3", "|", "Myr"))
    print("|" + "-"*w1 + "|" + "-"*w2 + "|")

    gc_data = []
    for n, gc_name in enumerate(clusters):
        try:
            o = Orbit.from_name(gc_name)
        except ValueError as e:
            print(str(e))
        else:
            print("|{:<12s}{:<12s}{:> 12.6f}{:> 12.6f}{:> 12.2f}{:> 12.2f}{:> 12.2f}{:> 12.2f}{:>12s}{:>12s}{:>15s}{:>6}{:^12s}|".format(
                "SIMBAD", o.name, o.ra(), o.dec(), o.dist(), o.vlos(), o.pmra(), o.pmdec(), "", "", "", "|", ""
            ))

        i, = numpy.where((H19_c["Cluster"] == gc_name.replace("Terzan", "Ter")))[0]
        gc_tcross = tcross(numpy.power(10, H19_c["rho_hm"][i]) * u.Msun/u.parsec**3)
        print("|{:<12s}{:<12s}{:> 12.6f}{:> 12.6f}{:> 12.2f}{:>12s}{:>12s}{:>12s}{:> 12.2e}{:> 12.2f}{:> 15.2f}{:>6}{:> 8.2f}    |".format(
            "H19_c", H19_c["Cluster"][i], H19_c["RA"][i], H19_c["DEC"][i], H19_c["R_Sun"][i], "", "", "",
            H19_c["Mass"][i], H19_c["rhm"][i],  H19_c["rho_hm"][i],
            "|", gc_tcross
        ))

        j, = numpy.where((B19["SimbadName"] == gc_name.replace("Terzan", "Ter").encode("utf-8")))[0]
        print("|{:<12s}{:<12s}{:> 12.6f}{:> 12.6f}{:> 12.2f}{:> 12.2f}{:> 12.2f}{:> 12.2f}{:>12s}{:>12s}{:>15s}{:>6}{:^12s}|".format(
            "B19", B19["SimbadName"][j].decode("ascii"), B19["RAJ2000"][j], B19["DEJ2000"][j],
            B19["Rsun"][j], B19["RV"][j], B19["pmRA_"][j], B19["pmDE"][j], "", "", "", "|", ""
        ))

        if n < len(clusters)-1: print("|" + " "*w1 + "|" + " "*w2 + "|")

        # Store the relevant GC data for future use
        # RA, Dec, distance (R_sun), radial velocity, proper motion RA, proper motion DEC from Baumgardt+ 2019
        # Mass, half-mass radius, density at the half-mass radius from the original reference
        # Baumgardt & Hilker (2018 --> 112 GCs), but that was supplemented up to 154 GCs by Hilker+ 2019
        # (conference prodeeding), for which the data lives on the HolgerBaumgardt website.
        # The crossing time is calulated based on the Hilker+ 2019 data
        gc_data.append([
            H19_c["Cluster"][i], H19_c["RA"][i], H19_c["DEC"][i], H19_c["R_Sun"][i],  # Baumgardt+ 2019
            B19["RV"][j], B19["pmRA_"][j], B19["pmDE"][j],  # Hilker+ 2019
            H19_c["Mass"][i], H19_c["rhm"][i],  H19_c["rho_hm"][i],  # Baumgardt+ 2019
            gc_tcross,  # calculated
        ])
    print("|" + "-"*w1 + "|" + "-"*w2 + "|")

    return gc_data


def plot_dispersions_and_Tdyn(H19_c, H19_rv, subset, debug=False):
    names = numpy.unique(numpy.array(
        [c.decode() for c in H19_rv["Cluster"]]
    ))
    print("There are {0} clusters /w velocity dispersion profiles".format(
        len(names)))

    for i, gc in enumerate(names):
        if gc not in subset:
            continue
        igc, = numpy.where(H19_rv["Cluster"] == gc.encode("utf-8"))
        radii = H19_rv["radius"][igc]
        velocity_dispersion = H19_rv["velocity_dispersion"][igc]
        velocity_dispersion_err_up = H19_rv["velocity_dispersion_err_up"][igc]
        velocity_dispersion_err_down = H19_rv["velocity_dispersion_err_down"][igc]

        reference = H19_rv["type"][igc]
        rv_from_h19, = numpy.where(reference == "RV".encode("utf-8"))  # Baumgardt, Hilker, Sollima & Bellini (2019), MNRAS 482, 5138
        if debug: print(rv_from_h19)
        rv_from_k18, = numpy.where(reference == "K18".encode("utf-8"))  # Kamann et al. (2018), MNRAS, 473, 5591
        if debug: print(rv_from_k18)
        pm_from_h19, = numpy.where(reference == "GDR2".encode("utf-8"))  # Baumgardt, Hilker, Sollima & Bellini (2019), MNRAS 482, 5138
        if debug: print(pm_from_h19)
        pm_from_w15, = numpy.where(reference == "W15".encode("utf-8"))  # Watkins et al. (2015), ApJ 803, 29
        if debug: print(pm_from_w15)

        fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(18, 9))
        # Radial velocity dispersion profiles
        ax1.errorbar(radii[rv_from_h19], velocity_dispersion[rv_from_h19],
            yerr=[velocity_dispersion_err_up[rv_from_h19], velocity_dispersion_err_down[rv_from_h19]],
            ls="none", marker="o", c="blue", ms=8, label="RV H19"
        )
        ax1.errorbar(radii[rv_from_k18], velocity_dispersion[rv_from_k18],
            yerr=[velocity_dispersion_err_up[rv_from_k18], velocity_dispersion_err_down[rv_from_k18]],
            ls="none", marker="^", c="purple", ms=8, label="RV K18"
        )

        # Proper motion dispersion profiles
        ax1.errorbar(radii[pm_from_h19], velocity_dispersion[pm_from_h19],
            yerr=[velocity_dispersion_err_up[pm_from_h19], velocity_dispersion_err_down[pm_from_h19]],
            ls="none", marker="o", c="r", ms=8, label="MP H19"
        )
        ax1.errorbar(radii[pm_from_w15], velocity_dispersion[pm_from_w15],
            yerr=[velocity_dispersion_err_up[pm_from_w15], velocity_dispersion_err_down[pm_from_w15]],
            ls="none", marker="o", c="orange", ms=8, label="PM W15"
        )

        ax1.text(0.95, 0.95, "{0}".format(gc), ha="right", va="top", transform=ax1.transAxes)
        ax1.set_xlabel("Radius [arcsec]")
        ax1.set_xscale("log")
        ax1.set_ylabel("Velocity dispersion [km/s]")
        ax1.legend(loc="lower left", fontsize=16, frameon=False)

        # Calculate Tdyn.  1 km/s ~ 1 parsec/Myr ~ 1 kpc/Gyr
        Tdyn = (radii*u.parsec/(velocity_dispersion*u.km/u.s)).to(u.Myr).value
        Tdyn_err_up = (radii*u.parsec/((velocity_dispersion+velocity_dispersion_err_up)*u.km/u.s)).to(u.Myr).value
        Tdyn_err_down = (radii*u.parsec/((velocity_dispersion-velocity_dispersion_err_down)*u.km/u.s)).to(u.Myr).value
        ax2.errorbar(radii, Tdyn, yerr=[abs(Tdyn-Tdyn_err_down), abs(Tdyn-Tdyn_err_up)],
            ls="none", marker="o", c="k", ms=8)

        iH19_c, = numpy.where(H19_c["Cluster"] == gc)[0]
        # Core radius using the Spitzer (1987) definition
        irc, = numpy.where(radii <= H19_c[iH19_c]["rc"])
        if len(irc) is 0: irc = [0]
        iTdyn_at_rc = irc[-1]
        ax2.axvline(H19_c[iH19_c]["rc"], c="red", ls=":", lw=2,
            label="core\n{0:.2f} Myr".format(Tdyn[iTdyn_at_rc]))
        # Projected half-light radius
        irhl, = numpy.where(radii <= H19_c[iH19_c]["rhl"])
        if len(irhl) is 0: irhl = [0]
        iTdyn_at_rhl = irhl[-1]
        ax2.axvline(H19_c[iH19_c]["rhl"], c="orange", ls=":", lw=2,
            label="half-light\n{0:6.2f} Myr".format(Tdyn[iTdyn_at_rhl]))
        # Half-mass radius (3D)
        irhl, = numpy.where(radii <= H19_c[iH19_c]["rhm"])
        iTdyn_at_rhm = irhl[-1]
        ax2.axvline(H19_c[iH19_c]["rhm"], c="green", ls=":", lw=2,
            label="half-mass (3D)\n{0:.2f} Myr".format(Tdyn[iTdyn_at_rhm]))
        # Tidal radius according to eq. 8 of Webb et al. (2013), ApJ 764, 124
        irt, = numpy.where(radii <= H19_c[iH19_c]["rt"])
        iTdyn_at_rt = irt[-1]
        ax2.axhline(Tdyn[iTdyn_at_rt], c="blue", lw=4)
        ax2.axvline(H19_c[iH19_c]["rt"], c="blue", ls=":", lw=2,
            label="tidal\n{0:.2f} Myr".format(Tdyn[iTdyn_at_rt]))

        ax2.set_xlabel("Radius [arcsec]")
        ax2.set_xscale("log")
        # ax2.set_yscale("log")
        ax2.set_ylabel(r"$\tau_{\rm dyn} = R/\sigma$ [Myr]")
        ax2.legend(fontsize=16, frameon=False)

        pyplot.tight_layout()
        pyplot.show()

    return names


def plot_radii(B19):
    fig, ax = pyplot.subplots(figsize=(12, 9))
    counts, edges = numpy.histogram(numpy.log10(B19["Rper"]), bins=12)
    ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid", c="k", lw=4)
    ax.set_xlabel("$\log_{10}$(Rper)")
    ax.axvline(numpy.log10(8), c="k", ls=":", lw=2)
    ax.set_ylabel("Count")
    pyplot.show()
    Rper_in_8kpc = len(numpy.where(B19["Rper"] < 8)[0])
    print("GCs /w Rper < 8 kpc: {0}".format(Rper_in_8kpc))
    print("GCs /w Rper >= 8 kpc: {0}".format(len(numpy.where(B19["Rper"] >= 8)[0])))
    print("{0:.2f}%".format(Rper_in_8kpc/len(B19["Rper"]) * 100))


    fig, ax = pyplot.subplots(figsize=(12, 9))
    counts, edges = numpy.histogram(numpy.log10(B19["Rapo"]), bins=12)
    ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid", c="k", lw=4)
    ax.set_xlabel("$\log_{10}$(Rapo)")
    ax.axvline(numpy.log10(8), c="k", ls=":", lw=2)
    ax.set_ylabel("Count")
    pyplot.show()
    Rapo_in_8kpc = len(numpy.where(B19["Rapo"] < 8)[0])
    # The 8 kpc line does not make sense for Rapo
    print("GCs /w Rapo < 8 kpc: {0}".format(Rapo_in_8kpc))
    print("GCs /w Rapo >= 8 kpc: {0}".format(len(numpy.where(B19["Rapo"] >= 8)[0])))
    print("{0:.2f}%".format(Rapo_in_8kpc/len(B19["Rapo"]) * 100))


    fig, ax = pyplot.subplots(figsize=(12, 9))
    counts, edges = numpy.histogram(numpy.log10(B19["Rsun"]), bins=12)
    ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid", c="k", lw=4)
    ax.set_xlabel("$\log_{10}$(Rsun)")
    ax.axvline(numpy.log10(8), c="k", ls=":", lw=2)
    ax.set_ylabel("Count")
    pyplot.show()
    Rsun_in_8kpc = len(numpy.where(B19["Rsun"] < 8)[0])
    # The 8 kpc line does not make sense for Rsun; it would for distance to center of the Galaxy
    print("GCs /w Rsun < 8 kpc: {0}".format(Rsun_in_8kpc))
    print("GCs /w Rsun >= 8 kpc: {0}".format(len(numpy.where(B19["Rsun"] >= 8)[0])))
    print("{0:.2f}%".format(Rsun_in_8kpc/len(B19["Rsun"]) * 100))


    fig, ax = pyplot.subplots(figsize=(12, 12))
    ax.plot(B19["Rper"], B19["Rapo"], "ro", ms=6)
    ax.axvline(8, c="k", ls=":", lw=2)
    ax.axhline(8, c="k", ls=":", lw=2)
    ax.set_xlabel("Rper")
    ax.set_ylabel("Rapo")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect(1)
    pyplot.show()


if __name__ == "__main__":

    # NGC 5139 ==> Omega Cen
    H19_c = parse_hilker_2019_combined(LOGGER)
    B19 = numpy.array(Vizier.get_catalogs('J/MNRAS/482/5138/table1')[0])
    CLUSTERS = ["NGC 104", "Whiting 1", "Pal 1", "NGC 5139", "Terzan 1", "NGC 7089"]
    CLUSTERS = numpy.intersect1d(
        H19_c["Cluster"],
        numpy.array(([c.decode() for c in B19["SimbadName"]]))
    )
    GC_DATA = print_hilker_vs_simbad(CLUSTERS, H19_c, B19)





    CLUSTERS_WITH_VELOCITY_DISPERSIONS = numpy.unique(numpy.array(
        [c.decode() for c in H19_velocity_dispersions["Cluster"]]
    ))
    print("There are {0} clusters /w velocity dispersion profiles".format(
        len(CLUSTERS_WITH_VELOCITY_DISPERSIONS)))

    for i, gc in enumerate(CLUSTERS_WITH_VELOCITY_DISPERSIONS):
        if gc not in ["NGC 104", "NGC 7089", "NGC 5139"]:
            continue
        igc, = numpy.where(H19_velocity_dispersions["Cluster"] == gc.encode("utf-8"))
        radii = H19_velocity_dispersions["radius"][igc]
        velocity_dispersion = H19_velocity_dispersions["velocity_dispersion"][igc]
        velocity_dispersion_err_up = H19_velocity_dispersions["velocity_dispersion_err_up"][igc]
        velocity_dispersion_err_down = H19_velocity_dispersions["velocity_dispersion_err_down"][igc]

        reference = H19_velocity_dispersions["type"][igc]
        rv_from_h19, = numpy.where(reference == "RV".encode("utf-8"))  # Baumgardt, Hilker, Sollima & Bellini (2019), MNRAS 482, 5138
        print(rv_from_h19)
        rv_from_k18, = numpy.where(reference == "K18".encode("utf-8"))  # Kamann et al. (2018), MNRAS, 473, 5591
        print(rv_from_k18)
        pm_from_h19, = numpy.where(reference == "GDR2".encode("utf-8"))  # Baumgardt, Hilker, Sollima & Bellini (2019), MNRAS 482, 5138
        print(pm_from_h19)
        pm_from_w15, = numpy.where(reference == "W15".encode("utf-8"))  # Watkins et al. (2015), ApJ 803, 29
        print(pm_from_w15)

        fig, ax = pyplot.subplots()
        # Radial velocity dispersion profiles
        ax.errorbar(radii[rv_from_h19], velocity_dispersion[rv_from_h19],
            yerr=[velocity_dispersion_err_up[rv_from_h19], velocity_dispersion_err_down[rv_from_h19]],
            ls="none", marker="o", c="blue", ms=8, label="RV H19"
        )
        ax.errorbar(radii[rv_from_k18], velocity_dispersion[rv_from_k18],
            yerr=[velocity_dispersion_err_up[rv_from_k18], velocity_dispersion_err_down[rv_from_k18]],
            ls="none", marker="^", c="purple", ms=8, label="RV K18"
        )

        # Proper motion dispersion profiles
        ax.errorbar(radii[pm_from_h19], velocity_dispersion[pm_from_h19],
            yerr=[velocity_dispersion_err_up[pm_from_h19], velocity_dispersion_err_down[pm_from_h19]],
            ls="none", marker="o", c="r", ms=8, label="MP H19"
        )
        ax.errorbar(radii[pm_from_w15], velocity_dispersion[pm_from_w15],
            yerr=[velocity_dispersion_err_up[pm_from_w15], velocity_dispersion_err_down[pm_from_w15]],
            ls="none", marker="o", c="orange", ms=8, label="PM W15"
        )

        ax.text(0.95, 0.95, "{0}".format(gc), ha="right", va="top", transform=ax.transAxes)
        ax.set_xlabel("Radius [parsec]")
        ax.set_xscale("log")
        ax.set_ylabel("Velocity dispersion [km/s]")
        ax.legend(loc="lower left", fontsize=16, frameon=False)
        pyplot.show()
