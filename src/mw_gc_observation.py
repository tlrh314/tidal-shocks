import os
import sys
import time
import pickle
import platform

import numpy
import scipy
from scipy import signal
import matplotlib
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
pyplot.style.use("tlrh")
from amuse.units import units

from limepy_wrapper import (
    king_wrapper,
    wilson_wrapper,
    limepy_wrapper,
    spes_wrapper
)
from emcee_wrapper import (
    minimise_chisq,
    run_mcmc,
    get_tau,
    get_flat_samples,
)
from amuse_wrapper import (
    get_radial_profiles,
    project_amuse_profiles,
    scatter_particles_xyz,
    print_particleset_info,
)
from galpy_amuse_wrapper import limepy_to_amuse

BASEDIR = "/u/timoh/phd" if "freya" in platform.node() else ""
if "/limepy" not in sys.path:
    sys.path.insert(0, "{}/limepy".format(BASEDIR))
import limepy   # using tlrh314/limepy fork

if "/supaharris" not in sys.path:
    sys.path.insert(0, "{}/supaharris".format(BASEDIR))
from utils import parsec2arcmin
from utils import arcmin2parsec
from data.parse_harris_1996ed2010 import parse_harris1996ed2010
from data.parse_balbinot_2018 import parse_balbinot_2018
from data.parse_hilker_2019 import parse_hilker_2019_orbits
from data.parse_hilker_2019 import parse_hilker_2019_combined
from data.parse_hilker_2019 import parse_hilker_2019_radial_velocities
from data.parse_deBoer_2019 import plot_deBoer_2019
from data.parse_deBoer_2019 import parse_deBoer_2019_fits
from data.parse_deBoer_2019 import parse_deBoer_2019_stitched_profiles


class MwGcObservation(object):
    def __init__(self, logger, gc_name, force_parse=False):
        self.logger = logger
        self.gc_name = gc_name
        self.gc_slug = self.gc_name.replace(" ", "").lower()
        self._set_outdir()
        self._set_observations(force_parse=force_parse)
        self.distance_kpc = self.h96_gc.dist_from_sun
        self.rJ_pc = self.b18["r_J"]
        self.rJ = parsec2arcmin(self.rJ_pc, self.distance_kpc)

        # Placeholder to store the fit data
        self.fit = {"king": {}, "wilson": {}, "limepy": {}, "spes": {}}

    def _set_outdir(self):
        self.outdir = "{}/tidalshocks/out/{}/".format(BASEDIR, self.gc_slug)
        if not os.path.exists(self.outdir) or not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
            self.logger.debug("  Created outdir {0}\n".format(self.outdir))
        else:
            self.logger.debug("  Using outdir: {0}\n".format(self.outdir))

    def _set_observations(self, force_parse=False):
        fname = "{}{}_obs_dump.p".format(self.outdir, self.gc_slug)
        print(fname)

        # Load dump of all observations if it exists, else parse underlying data
        if os.path.exists(fname) and os.path.isfile(fname) and not force_parse:
            p = pickle.load(open(fname, "rb"))
            self.h96_gc = p["h96_gc"]
            self.b18 = p["b18"]
            self.h19_orbit = p["h19_orbit"]
            self.h19_combined = p["h19_combined"]
            self.h19_rv = p["h19_rv"]
            self.deB19_fit = p["deB19_fit"]
            self.deB19_stitched = p["deB19_stitched"]
        else:
            # Various parameters
            self._set_harris1996()

            # Jacobi radii
            self._set_balbinot2018()

            # Projected star count profiles
            self._set_deBoer2019()

            # Orbital parameters for 154 MW GCs, and velocity dispersion profiles
            self._set_hilker2019()

            pickle.dump({
                    "h96_gc": self.h96_gc,
                    "b18": self.b18,
                    "h19_orbit": self.h19_orbit,
                    "h19_combined": self.h19_combined,
                    "h19_rv": self.h19_rv,
                    "deB19_fit": self.deB19_fit,
                    "deB19_stitched": self.deB19_stitched,
                }, open(fname, "wb")
            )

    def _set_harris1996(self):
        self.h96_gc = parse_harris1996ed2010(self.logger)[self.gc_name]

    def _set_balbinot2018(self):
        b18 = parse_balbinot_2018(self.logger)
        imatch, = numpy.where(b18["Name"] == self.gc_name)[0]
        self.b18 = b18[imatch]

    def _set_hilker2019(self):
        h19_orbits = parse_hilker_2019_orbits(self.logger)
        imatch, = numpy.where(h19_orbits["Cluster"] == self.gc_name)[0]
        self.h19_orbit = h19_orbits[imatch]

        h19_combined = parse_hilker_2019_combined(self.logger)
        imatch, = numpy.where(h19_combined["Cluster"] == self.gc_name)[0]
        self.h19_combined = h19_combined[imatch]

        h19_rvs = parse_hilker_2019_radial_velocities(self.logger)
        imatch, = numpy.where(h19_rvs["Cluster"] == self.gc_name)
        self.h19_rv = h19_rvs[imatch]

    def _set_deBoer2019(self):
        # de Boer+ (2019): star count as function of radius
        deB19_fits = parse_deBoer_2019_fits(self.logger)
        deB19_stitched = parse_deBoer_2019_stitched_profiles(self.logger)
        self.deB19_fit = deB19_fits[
            numpy.where(deB19_fits["id"] == self.gc_name)[0][0]
        ]
        self.deB19_stitched = deB19_stitched[self.gc_name]

    def add_H19_RVs_to_fig(self, fig):
        # Radial velocity dispersion from 2018MNRAS.473.5591K
        rv_k18, = numpy.where(self.h19_rv["type"] == "K18")
        # Radial velocity dispersion from 2019MNRAS.482.5138B
        rv_h19, = numpy.where(self.h19_rv["type"] == "RV")

        # Proper motion dispersion from 2015ApJ...803...29W
        pm_w15, = numpy.where(self.h19_rv["type"] == "W15")
        # Proper motion dispersion from 2019MNRAS.482.5138B
        pm_h19, = numpy.where(self.h19_rv["type"] == "GDR2")

        ax = pyplot.gca()
        all_radii = arcmin2parsec(self.h19_rv["radius"]/60, self.distance_kpc)
        for cut, label, c in zip([rv_k18, rv_h19, pm_w15, pm_h19],
                ["RV K18 (MUSE)", "RV H19", "PM W15 (HST)", "PM H19 (Gaia)"],
                ["red", "blue", "green", "orange"]):
            radii = all_radii[cut]
            ax.errorbar(radii,
                self.h19_rv[cut]["velocity_dispersion"], yerr=[
                    self.h19_rv[cut]["velocity_dispersion_err_down"],
                    self.h19_rv[cut]["velocity_dispersion_err_up"] ],
                marker="o", c=c, ls="", ms=4, elinewidth=2,
                markeredgewidth=2, capsize=5, label=label
            )
        ax.set_xlim(0.01*numpy.min(all_radii), 10*numpy.max(all_radii))
        ax.set_xscale("log")
        ax.set_xlabel("Radius [parsec]")
        ax.set_ylabel("$\sigma_{1D}$ [km/s]")
        ax.legend(loc="lower left", fontsize=16, frameon=False)

    def fit_model_to_deBoer2019(self, model="king",
            mcmc=True, Nwalkers=32, Nsamples=500, progress=True,
            mask_2BGlev=False, mask_rtie=False, verbose=False):
        self.fit_x = self.deB19_stitched["rad"]
        self.fit_y = self.deB19_stitched["density"]
        self.fit_yerr = self.deB19_stitched["density_err"]

        # Mask the data / take a sub set of the data into account
        if mask_2BGlev:
            # Only include data points above 2 times the background level into the fit
            ikeep, = numpy.where(self.fit_y > 2*self.deB19_fit["BGlev"])
        if mask_rtie:
            # Only include data points at radii below the Gaia completeness radius
            ikeep, = numpy.where(self.fit_x < self.deB19_fit["r_tie"])
        if mask_2BGlev or mask_rtie:
            self.fit_x = self.fit_x[ikeep]
            self.fit_y = self.fit_y[ikeep]
            self.fit_yerr = self.fit_yerr[ikeep]

        # if model == "woolley": --> g=0
        if model == "king":
            W0_deB19_king = self.deB19_fit["W_king"]
            M_deB19_king = self.deB19_fit["M_king"]
            rt_deB19_king = parsec2arcmin(self.deB19_fit["rt_king"], self.distance_kpc)
            self.fit["king"]["initial"] = [W0_deB19_king, M_deB19_king, rt_deB19_king]
            self.fit["king"]["deB19_chi2"] = self.deB19_fit["chi2_king"]
            self.fit["king"]["fit_labels"] = ["W_0", "M", "r_t"]
            self.fit["king"]["model"] = king_wrapper
        elif model == "wilson":
            W0_deB19_wilson = self.deB19_fit["W_wil"]
            M_deB19_wilson = self.deB19_fit["M_wil"]
            rt_deB19_wilson = parsec2arcmin(self.deB19_fit["rt_wil"], self.distance_kpc)
            self.fit["wilson"]["initial"] = [W0_deB19_wilson, M_deB19_wilson, rt_deB19_wilson]
            self.fit["wilson"]["deB19_chi2"] = self.deB19_fit["chi2_wil"]
            self.fit["wilson"]["fit_labels"] = ["W_0", "M", "r_t"]
            self.fit["wilson"]["model"] = wilson_wrapper
        elif model == "limepy":
            W0_deB19_lime = self.deB19_fit["W_lime"]
            M_deB19_lime = self.deB19_fit["M_lime"]
            rt_deB19_lime = parsec2arcmin(self.deB19_fit["rt_lime"], self.distance_kpc)
            g_deB19_lime = self.deB19_fit["g_lime"]
            self.fit["limepy"]["initial"] = [W0_deB19_lime, M_deB19_lime, rt_deB19_lime, g_deB19_lime]
            self.fit["limepy"]["deB19_chi2"] = self.deB19_fit["chi2_lime"]
            self.fit["limepy"]["fit_labels"] = ["W_0", "M", "r_t", "g"]
            self.fit["limepy"]["model"] = limepy_wrapper
        elif model == "spes":
            W0_deB19_spes = self.deB19_fit["W_pe"]
            B_deB19_spes = 1 - numpy.power(10, self.deB19_fit["log1minB_pe"])
            eta_deB19_spes = self.deB19_fit["eta_pe"]
            M_deB19_spes = self.deB19_fit["M_pe"]
            rt_deB19_spes = parsec2arcmin(self.deB19_fit["rt_pe"], self.distance_kpc)
            # fpe_deB19_spes = numpy.power(10, self.deB19_fit["log_fpe"])
            self.fit["spes"]["initial"] = [W0_deB19_spes, B_deB19_spes,
                    eta_deB19_spes, M_deB19_spes, rt_deB19_spes]
            self.fit["spes"]["deB19_chi2"] = self.deB19_fit["chi2_pe"]
            self.fit["spes"]["fit_labels"] = ["W_0", "B", "eta", "M", "r_t"]
            self.fit["spes"]["model"] = lambda x, W0, B, eta, M, rt: spes_wrapper(
                x, W0, B, eta, M, rt, 25*self.rJ/rt)

        # Minimise chi^2
        if verbose: start = time.time()
        self.fit[model]["soln"] = minimise_chisq(self.fit[model]["initial"],
            self.fit_x, self.fit_y, self.fit_yerr, self.fit[model]["model"]
        )
        # chisq = self.fit[model]["soln"].fun
        if verbose:
            self.logger.info("minimise_chisq took {:.2f} seconds".format(
                time.time() - start))
            self.logger.info("Maximum likelihood estimates for minimise_chisq:")
            for label, mle in zip(self.fit[model]["fit_labels"], self.fit[model]["soln"].x):
                self.logger.info("  {} = {:.3f}".format(label, mle))
            self.logger.info("")
            self.logger.info("minimise_chisq full return\n{}".format(self.fit[model]["soln"]))

        if mcmc:
            self.fit[model]["mcmc_mle"] = []
            self.fit[model]["mcmc_err_up"] = []
            self.fit[model]["mcmc_err_down"] = []
            start = time.time()
            self.fit[model]["sampler"] = run_mcmc(self.fit[model]["soln"].x,
                self.fit_x, self.fit_y, self.fit_yerr, self.fit[model]["model"],
                self.outdir, Nwalkers=Nwalkers, Nsamples=Nsamples, progress=progress)
            if verbose:
                self.logger.info("run_mcmc took {:.2f} seconds".format(
                    time.time() - start))
            self.fit[model]["tau"] = get_tau(self.fit[model]["sampler"])
            self.fit[model]["flat_samples"] = \
                get_flat_samples(self.fit[model]["sampler"], self.fit[model]["tau"])

            ndim = len(self.fit[model]["initial"])
            self.fit[model]["mcmc_mle"] = []
            self.fit[model]["mcmc_err_down"] = []
            self.fit[model]["mcmc_err_up"] = []
            for i in range(ndim):
                mcmc = numpy.percentile(self.fit[model]["flat_samples"][:, i], [16, 50, 84])
                q = numpy.diff(mcmc)
                self.fit[model]["mcmc_mle"].append(mcmc[1])
                self.fit[model]["mcmc_err_down"].append(q[0])
                self.fit[model]["mcmc_err_up"].append(q[1])

    def add_deBoer2019_to_fig(self, fig,
            show_King=False, show_Wilson=False, show_limepy=False, show_spes=False,
            show_BGlev=True, show_rtie=True, show_rJ=True,
            has_tex=False, verbose=False):
        return plot_deBoer_2019(
            self.logger, self.deB19_fit, self.deB19_stitched,
            self.distance_kpc, self.rJ_pc, self.rJ, fig=fig,
            show_King=show_King, show_Wilson=show_Wilson,
            show_limepy=show_limepy, show_spes=show_spes,
            show_BGlev=show_BGlev, show_rtie=show_rtie, show_rJ=show_rJ,
            has_tex=has_tex, verbose=verbose
        )

    def sample_deBoer2019_bestfit_king(self, Nstars=1337, verbose=False):
        W0_deB19 = self.deB19_fit["W_king"]
        M_deB19 = self.deB19_fit["M_king"]
        rt_deB19 = parsec2arcmin(self.deB19_fit["rt_king"], self.distance_kpc)
        return limepy_to_amuse(W0_deB19, M=M_deB19, rt=rt_deB19, g=1,
            Nstars=Nstars, verbose=verbose)

    def sample_deBoer2019_bestfit_wilson(self, Nstars=1337, verbose=False):
        W0_deB19 = self.deB19_fit["W_wil"]
        M_deB19 = self.deB19_fit["M_wil"]
        rt_deB19 = parsec2arcmin(self.deB19_fit["rt_wil"], self.distance_kpc)
        return limepy_to_amuse(W0_deB19, M=M_deB19, rt=rt_deB19, g=2,
            Nstars=Nstars, verbose=verbose)

    def sample_deBoer2019_bestfit_limepy(self, Nstars=1337, verbose=False):
        W0_deB19 = self.deB19_fit["W_lime"]
        g_deB19 = self.deB19_fit["g_lime"]
        M_deB19 = self.deB19_fit["M_lime"]
        rt_deB19 = parsec2arcmin(self.deB19_fit["rt_lime"], self.distance_kpc)
        return limepy_to_amuse(W0_deB19, M=M_deB19, rt=rt_deB19, g=g_deB19,
            Nstars=Nstars, verbose=verbose)

    def add_deBoer2019_sampled_to_ax(self, ax, sampled, parm="rho", model=None,
            rmin=1e-3, rmax=1e3, Nbins=256, smooth=False, timing=True):
        if parm not in ["rho", "Sigma", "mc"]:
            self.logger.error("ERROR: cannot add {0} to ax".format(parm))
            return

        start = time.time()
        radii, N_in_shell, M_below_r, rho_of_r, volume = get_radial_profiles(
            sampled, c=sampled.center_of_mass().value_in(units.parsec),
            rmin=rmin, rmax=rmax, Nbins=Nbins, timing=timing)
        if timing:
            print("get_radial_profiles took {0:.2f} s".format(time.time() - start))

        if parm == "rho":
            if model is not None: ax.plot(model.r, model.rho)
            rho_plot = rho_of_r.value_in(units.MSun/units.parsec**3)
            # if smooth:
            #     rho_plot = scipy.signal.savgol_filter(rho_of_r, 21, 2)
            ax.plot(radii.value_in(units.parsec), rho_plot,
                c="r", lw=2, drawstyle="steps-mid", label=r"sampled $\rho(r)$"
            )
        elif parm == "Sigma":
            R, Sigma = project_amuse_profiles(radii, rho_of_r, timing=timing)
            ax.plot(R, Sigma, c="magenta", lw=2,
                drawstyle="steps-mid", label=r"sampled $\Sigma(R)$")
        elif parm == "mc":
            if model is not None: ax.plot(model.r, model.mc)
            ax.plot(radii.value_in(units.parsec),
                M_below_r.value_in(units.MSun),
                c="r", lw=2, drawstyle="steps-mid", label=r"sampled $M(<r)$")
            ax.set_xlim(0.9*radii.value_in(units.parsec).min(),
                1.1*radii.value_in(units.parsec).max())
            ax.set_ylim(0.2, 3*numpy.max(M_below_r.value_in(units.MSun)))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Radius [parsec]")
            ax.set_ylabel("Mass (< r) [MSun]")

        # Add characteristic radii
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        font = FontProperties()
        font.set_size(12)
        font.set_weight("bold")
        if model is not None:
            for n, r, c in zip(["King core", "half-mass", "virial", "truncation"],
                    [model.r0, model.rh, model.rv, model.rt],
                    # ["#980043", "#dd1c77", "#df65b0", "#d7b5d8"]
                    ["orange", "green", "blue", "red"]
            ):
                ax.vlines(r, ymin=self.deB19_fit["BGlev"]/25,
                    ymax=self.deB19_fit["BGlev"], color=c, lw=4)
                # ax.axvline(r, c=c, ls="-", lw=2)
                # ax.text(1.05*r, 0.98, "{0}: {1:.3f}".format(n, r), c=c, rotation=90,
                #         fontsize=12, ha="left", va="top", transform=trans)  #, fontproperties=font)

        # TODO: Add force softening
        xlim = ax.get_xlim()
        softening = parsec2arcmin(0.1, self.distance_kpc)
        ax.fill_between(numpy.arange(xlim[0], softening, 0.01), 0, 1,
            facecolor="grey", edgecolor="grey", alpha=0.2, transform=trans)


    def __str__(self):
        s = "MwGcObservation for {0}\n".format(self.gc_name)
        s += "  R_sun: {0:.2f} kpc (Harris 1996, 2010 ed.)\n".format(
            self.distance_kpc)
        s += "  rJ:    {0:.2f} pc (Balbinot & Gieles 2018) --> {1:.2f}'\n".format(
            self.rJ_pc, self.rJ)
        return s


if __name__ == "__main__":
    import logging
    logging.getLogger("keyring").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(message)s")
    logger = logging.getLogger(__file__)
    logger.info("Running {0}".format(__file__))

    obs = MwGcObservation(logger, "NGC 104")
    logger.info(obs)

    fig, ax = pyplot.subplots(1, 1, figsize=(12, 12))
    pyplot.switch_backend("TkAgg")
    obs.add_deBoer2019_to_fig(fig)
    pyplot.show()
