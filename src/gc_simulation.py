import os
import sys
import time
import copy
import logging
import platform

import numpy
import scipy
from matplotlib import pyplot
pyplot.style.use("tlrh")
from amuse.units import units

from tlrh_profiles import (
    limepy_wrapper,
    spes_wrapper,
    minimise_chisq,
    log_likelihood,
    run_mcmc,
    inspect_chains,
    get_tau,
    get_flat_samples,
)
from galpy_amuse import limepy_to_amuse
from tlrh_datamodel import get_radial_profiles
from tlrh_datamodel import scatter_particles_xyz

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


class StarClusterSimulation(object):
    def __init__(self, logger, gc_name):
        self.logger = logger
        self.gc_name = gc_name
        self._set_outdir()

        self._set_observations()

    def _set_outdir(self):
        self.outdir = "{}/tidalshocks/out/{}/".format(BASEDIR, self.gc_name)
        if not os.path.exists(self.outdir) or not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
            self.logger.debug("  Created outdir {0}\n".format(self.outdir))
        else:
            self.logger.debug("  Using outdir: {0}\n".format(self.outdir))

    def _set_observations(self):
        # Various parameters
        self._set_harris1996()
        self.distance_kpc = self.h96_gc.dist_from_sun

        # Jacobi radii
        self._set_balbinot2018()
        self.rJ_pc = self.b18["r_J"]
        self.rJ = parsec2arcmin(self.rJ_pc, self.distance_kpc)

        # Projected star count profiles
        self._set_deBoer2019()

        # Orbital parameters for 154 MW GCs, and velocity dispersion profiles
        self._set_hilker2019()

        # Placeholder to store the fit data
        self.fit = {"king": {}, "wilson": {}, "limepy": {}, "spes": {}}

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
            self.fit["king"]["model"] = lambda x, W0, M, rt: limepy_wrapper(x, W0, M, rt, g=1)
        elif model == "wilson":
            W0_deB19_wilson = self.deB19_fit["W_wil"]
            M_deB19_wilson = self.deB19_fit["M_wil"]
            rt_deB19_wilson = parsec2arcmin(self.deB19_fit["rt_wil"], self.distance_kpc)
            self.fit["wilson"]["initial"] = [W0_deB19_wilson, M_deB19_wilson, rt_deB19_wilson]
            self.fit["wilson"]["deB19_chi2"] = self.deB19_fit["chi2_wil"]
            self.fit["wilson"]["fit_labels"] = ["W_0", "M", "r_t"]
            self.fit["wilson"]["model"] = lambda x, W0, M, rt: limepy_wrapper(x, W0, M, rt, g=2)
        elif model == "limepy":
            W0_deB19_lime = self.deB19_fit["W_lime"]
            M_deB19_lime = self.deB19_fit["M_lime"]
            rt_deB19_lime = parsec2arcmin(self.deB19_fit["rt_lime"], self.distance_kpc)
            g_deB19_lime = self.deB19_fit["g_lime"]
            self.fit["limepy"]["initial"] = [W0_deB19_lime, M_deB19_lime, rt_deB19_lime, g_deB19_lime]
            self.fit["limepy"]["deB19_chi2"] = self.deB19_fit["chi2_lime"]
            self.fit["limepy"]["fit_labels"] = ["W_0", "M", "r_t", "g"]
            self.fit["limepy"]["model"] = lambda x, W0, M, rt, g: limepy_wrapper(x, W0, M, rt, g=g)  # g free
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
        ding = -log_likelihood(self.fit[model]["soln"].x, self.fit_x,
                self.fit_y, self.fit_yerr, self.fit[model]["model"])
        print(ding)
        if verbose:
            self.logger.info("minimise_chisq took {:.2f} seconds".format(
                time.time() - start))
            self.logger.info("Maximum likelihood estimates for minimise_chisq:")
            for label, mle in zip(self.fit[model]["fit_labels"], self.fit[model]["soln"].x):
                self.logger.info("  {} = {:.3f}".format(label, mle))
            self.logger.info("")

        if mcmc:
            self.fit[model]["mcmc_mle"] = []
            self.fit[model]["mcmc_err_up"] = []
            self.fit[model]["mcmc_err_down"] = []
            start = time.time()
            self.fit[model]["sampler"] = run_mcmc(self.fit[model]["soln"].x,
                self.fit_x, self.fit_y, self.fit_yerr, self.fit[model]["model"],
                Nwalkers=Nwalkers, Nsamples=Nsamples, progress=progress)
            if verbose:
                self.logger.info("run_mcmc took {:.2f} seconds".format(
                    time.time() - start))
            # fig = inspect_chains(self.fit[model]["sampler"], self.fit[model]["fit_labels"])
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
        self.king_model, self.king_limepy_sampled, self.king_amuse, self.converter = \
            limepy_to_amuse(W0_deB19, M=M_deB19, rt=rt_deB19, g=1,
            Nstars=Nstars, verbose=verbose
        )

    def _set_amuse_radial_profiles(self, rmin, rmax, Nbins, verbose=False):
        if not hasattr(self, "king_amuse"):
            self.logger.error("Cannot add sampled King profile to fig")
            return
        (self.amuse_radii, self.amuse_N_in_shell, self.amuse_M_below_r,
         self.amuse_rho_of_r, self.amuse_volume) = get_radial_profiles(
            self.king_amuse, rmin=rmin, rmax=rmax, Nbins=Nbins, verbose=verbose)

    def _project_amuse(self):
        # Shamelessly copied from mgieles/limepy/limepy.py, but see
        # 2015MNRAS.454..576G eq. 35
        R = copy.copy(self.amuse_radii.value_in(units.parsec))
        radii = self.amuse_radii.value_in(units.parsec)
        rho = self.amuse_rho_of_r.value_in(units.MSun/units.parsec**3)
        Sigma = numpy.zeros(len(self.amuse_radii))
        for i in range(len(self.amuse_radii)-1):
            c = (radii >= R[i])
            r = radii[c]
            z = numpy.sqrt(abs(r**2 - R[i]**2)) # avoid small neg. values
            Sigma[i] = 2.0*abs(scipy.integrate.simps(rho[c], x=z))
        self.amuse_R = R
        self.amuse_Sigma = Sigma

        # TODO: Velocity dispersion profiles

    def add_deBoer2019_sampled_to_ax(self, ax, parm="rho",
            rmin=1e-3, rmax=1e3, Nbins=256):
        if parm not in ["rho", "Sigma", "mc"]:
            self.logger.error("ERROR: cannot add {0} to ax".format(parm))
            return
        self._set_amuse_radial_profiles(rmin, rmax, Nbins)

        # iNstar = numpy.where(self.amuse_N_in_shell == len(self.king_amuse.x))[0][0]
        # self.amuse_rt = self.amuse_radii[iNstar].value_in(units.parsec)
        # # TODO: get the truncation radius from the sampled profile. Seems to differ ~5%
        # # from the 'true' value that we know from the underlying limepy model that we sample
        # # print(self.amuse_rt, self.king_model.rt)
        # ax.axvline(self.amuse_rt, c="r", lw=4)

        if parm == "rho":
            ax.plot(self.king_model.r, self.king_model.rho)
            ax.plot(self.amuse_radii.value_in(units.parsec),
                self.amuse_rho_of_r.value_in(units.MSun/units.parsec**3),
                c="r", lw=2, drawstyle="steps-mid", label=r"sampled $\rho(r)$"
            )
        elif parm == "Sigma":
            self._project_amuse()
            ax.plot(self.amuse_R, self.amuse_Sigma, c="magenta", lw=2,
                drawstyle="steps-mid", label=r"sampled $\Sigma(R)$")
        elif parm == "mc":
            ax.plot(self.king_model.r, self.king_model.mc)
            ax.plot(self.amuse_radii.value_in(units.parsec),
                self.amuse_M_below_r.value_in(units.MSun),
                c="r", lw=2, drawstyle="steps-mid", label=r"sampled $M(<r)$")
            ax.set_xlim(0.9*self.amuse_radii.value_in(units.parsec).min(),
                1.1*self.amuse_radii.value_in(units.parsec).max())
            ax.set_ylim(0.2, 3*numpy.max(self.amuse_M_below_r.value_in(units.MSun)))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Radius [parsec]")
            ax.set_ylabel("Mass (< r) [MSun]")

    def analyse_isolation(self, rmin=1e-3, rmax=1e3, Nbins=256):
        import glob
        from amuse.units import units
        from amuse.io import read_set_from_file
        from tlrh_datamodel import print_particleset_info
        snap_base = "{}{}_isolation_*.hdf5".format(self.outdir, self.gc_name)
        snapshots = glob.glob(snap_base)
        print("Found {0} snapshots".format(len(snapshots)))

        # Initial conditions
        # Plot Sigma(R) at this time step
        fig, ax = pyplot.subplots(1, 1, figsize=(12, 9))
        self.add_deBoer2019_to_fig(fig, show_King=True)
        print(self.king_amuse.total_mass().value_in(units.MSun))
        self.add_deBoer2019_sampled_to_ax(ax, parm="Sigma")
        ax.legend(fontsize=20)
        pyplot.savefig("{0}{1}_isolation_ICs.png".format(self.outdir, self.gc_name))
        pyplot.show(fig)

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
            print_particleset_info(stars, self.converter, modelname)

            self.king_amuse = stars
            self._set_amuse_radial_profiles(rmin, rmax, Nbins)
            self._project_amuse()

            fig = scatter_particles_xyz(self.king_amuse)
            pyplot.show(fig)

            # Plot Sigma(R) at this time step
            fig, ax = pyplot.subplots(1, 1, figsize=(12, 9))
            self.add_deBoer2019_to_fig(fig, show_King=True)
            self.add_deBoer2019_sampled_to_ax(ax, parm="Sigma")
            ax.legend(fontsize=20)
            pyplot.savefig("{0}{1}_isolation_{2:04d}.png".format(self.outdir, self.gc_name, i))
            pyplot.show(fig)


    def __str__(self):
        s = "StarClusterSimulation for {0}\n".format(self.gc_name)
        s += "  R_sun: {0:.2f} kpc (Harris 1996, 2010 ed.)\n".format(
            self.distance_kpc)
        s += "  rJ:    {0:.2f} pc (Balbinot & Gieles 2018) --> {1:.2f}'\n".format(
            self.rJ_pc, self.rJ)
        s += "\n"
        s += "  outdir: {}".format(self.outdir)
        s += "\n"
        return s


if __name__ == "__main__":
    logging.getLogger("keyring").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(message)s")
    logger = logging.getLogger(__file__)
    logger.info("Running {0}".format(__file__))

    sim = StarClusterSimulation(logger, "NGC 104")
    logger.info(sim)

    fig, ax = pyplot.subplots(1, 1, figsize=(10, 10))
    pyplot.switch_backend("TkAgg")
    sim.add_deBoer2019_to_fig(fig)
    pyplot.show()
