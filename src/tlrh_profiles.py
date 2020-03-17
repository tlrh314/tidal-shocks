import re
import sys
import time
import argparse
import platform

import numpy
import scipy
from scipy.optimize import minimize
from matplotlib import pyplot
pyplot.style.use("tlrh")

# Some builds of NumPy (including the version included with Anaconda) will
# automatically parallelize some operations using something like the MKL
# linear algebra. This can cause problems when used with the parallelization
# methods described here so it can be good to turn that off (by setting the
# environment variable OMP_NUM_THREADS=1, for example).
import os
os.environ["OMP_NUM_THREADS"] = "1"

import emcee
from emcee.autocorr import AutocorrError
from multiprocessing import Pool
from multiprocessing import cpu_count
# print("{0} CPUs".format(cpu_count()))

BASEDIR = "/u/timoh/phd/" if "freya" in platform.node() else ""
if "/limepy" not in sys.path:
    sys.path.insert(0, "{}/limepy".format(BASEDIR))
import limepy   # using tlrh314/limepy fork


def limepy_wrapper(x, W0, M, rt, g=1, verbose=True):
    if verbose:
        print("Limepy, W0={0:.3f}, M={1:.3e}, rt={2:.3f}, g={3:.3f}".format(
            W0, M, rt, g))
    lime = limepy.limepy(W0, M=M, rt=rt, g=g, project=True, verbose=False)

    # Interpolate the projected surface density profile to allow for arbitrary radii
    interp1d = scipy.interpolate.interp1d(lime.R, lime.Sigma)

    # However, interp1d needs to stay within bounds. The Limepy profile is
    # only sampled up to the truncation radius rt, and cannot be interpolated
    # beyond. The surface brightness profile, by construction, is zero at the
    # truncation radius. So we set the counts to ZERO equal to small value
    ZERO = 1e-9
    counts = numpy.array([interp1d(xi) if xi < lime.rt else ZERO for xi in x])

    return counts


def spes_wrapper(x, W0, B, eta, M, rt, nrt, verbose=True):
    # Prevent infinite loop when spes has B or eta out of bounds
    if (0.0 > B or B > 1.0) or (0.0 > eta or eta > 1.0): return numpy.ones(len(x))

    if verbose:
        print("Spes, W0={0:.3f}, B={1:.3f}, eta={2:.3f}, M={3:.3e}, rt={4:.3f}, nrt={5:.3f}".format(
            W0, B, eta, M, rt, nrt))
    start = time.time()
    spes = limepy.spes(W0, B=B, eta=eta, M=M, rt=rt, nrt=nrt, project=True, verbose=False)
    print("spes_wrapper took {:.2f} seconds".format(time.time() - start))

    # Interpolate the projected surface density profile to allow for arbitrary radii
    interp1d = scipy.interpolate.interp1d(spes.R, spes.Sigma)

    # However, interp1d needs to stay within bounds. The Limepy profile is
    # only sampled up to the truncation radius rt, and cannot be interpolated
    # beyond. The surface brightness profile, by construction, is zero at the
    # truncation radius. So we set the counts to ZERO equal to small value
    ZERO = 1e-9
    counts = numpy.array([interp1d(xi) if xi < spes.rt else ZERO for xi in x])

    return counts


def log_likelihood(theta, x, y, yerr, model):
    ymodel = model(x, *theta)
    sigma2 = yerr**2
    # Note that I added 2*numpy.pi in the logarithm of sigma2 wrt documentation
    return -0.5 * numpy.sum((y - ymodel)**2 / sigma2 + numpy.log(2*numpy.pi*sigma2))


def log_prior(theta):
    if len(theta) == 3:  # King or Wilson
        W0, M, rt = theta
        if 0 < W0 < 14 and 1e2 < M < 1e6 and 1 < rt < 300:
            return 0.0
    elif len(theta) == 4:  # Limepy
        W0, M, rt, g = theta
        if 0 < W0 < 14 and 1e2 < M < 1e6 and 1 < rt < 300 and 0 < g < 3.5:
            return 0.0
    elif len(theta) == 5:  # SPES
        W0, B, eta, M, rt = theta
        if 0 < W0 < 14 and 0 < B < 1 and 0 < eta < 1 and 1e2 < M < 1e6 and 1 < rt < 300:
            return 0.0
    # TODO: why 0.0 if parameter in range, and -inf if parameter not in range?
    # The documentation shows that p(m) is 1/5.5 if -5 < m < 1/2, but this
    # method returns 0.0. That does not make much sense, does it?
    # TODO: may want lognormal weak priors rather than this business of
    # "uninformative flat priors"
    return -numpy.inf


def log_probability(theta, x, y, yerr, model):
    lp = log_prior(theta)
    if not numpy.isfinite(lp):
        return -numpy.inf
    return lp + log_likelihood(theta, x, y, yerr, model)


def minimise_chisq(initial, x, y, yerr, model):
    negative_log_likelihood = lambda *args: -log_likelihood(*args)
    return minimize(negative_log_likelihood, initial, args=(x, y, yerr, model))


def run_mcmc(ml, x, y, yerr, model, Nwalkers=32, Nsamples=500, progress=True):
    Ndim = len(ml)
    pos = ml + 1e-4 * numpy.random.randn(Nwalkers, Ndim)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(Nwalkers, Ndim,
            log_probability, args=(x, y, yerr, model), pool=pool)
        sampler.run_mcmc(pos, Nsamples, progress=progress);
    return sampler


def inspect_chains(sampler, labels):
    samples = sampler.get_chain()
    ndim = samples.shape[-1]
    fig, axes = pyplot.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    return fig


def get_tau(sampler):
    try:
        tau = sampler.get_autocorr_time()
    except AutocorrError as e:
        print(e)
        tau_list = str(e)[str(e).find("tau")+5:]
        tau_list_clean = re.sub("\s+", ",", tau_list.replace("[  ", "").replace(
            "[ ", "").replace("[", "").replace("]", ""))
        tau = numpy.fromstring(tau_list_clean, sep=",")
    print("tau: {0}".format(tau))
    print(tau.shape, tau.dtype)
    return tau


def get_flat_samples(sampler, tau, discard=None, thin=None):
    mean_tau = int(numpy.ceil(numpy.mean(tau)))
    if discard is None:
        discard = int(numpy.ceil(mean_tau))
    if thin is None:
        thin = int(numpy.ceil(mean_tau/2))
    print("discard: {0}, thin: {1}".format(discard, thin))
    return sampler.get_chain(discard=discard, thin=thin, flat=True)


def plot_corner(flat_samples, truths, labels):
    import corner

    # Use the best-fit MLEs from deBoer+ 2019 as 'truths'
    fig, axes = pyplot.subplots(3, 3, figsize=(16, 16), facecolor="white")
    fig = corner.corner(
        flat_samples, bins=64, labels=labels, show_titles=True,
        truths=truths, plot_contours=False,
        quantiles=[0.16, 0.50, 0.84], top_ticks=True, fig=fig, verbose=True
    )
    return fig


def plot_results(sim, models=["king", "wilson", "limepy", "spes"],
        x0=numpy.logspace(-3, 3, 256)):
    fig, ax = pyplot.subplots(1, 1, figsize=(12, 9))
    sim.add_deBoer2019_to_fig(fig)
    ax.errorbar(sim.fit_x, sim.fit_y, yerr=sim.fit_yerr, fmt="ro",
        ms=4, capsize=0, label="Included in fit")
    kwargs_deB19 = {
        "king": {
            "label": "King (deB19), chi2={0:.1f}",
            "c": "b", "ls": ":",  "lw": 2,
        }, "wilson": {
            "label": "Wilson (deB19), chi2={0:.1f}",
            "c": "g", "ls": "-.", "lw": 2,
        }, "limepy": {
            "label": "Limepy (deB19), chi2={0:.1f}",
            "c": "k", "ls": "--", "lw": 2,
        }, "spes": {
            "label": "SPES (deB19), chi2={0:.1f}",
            "c": "r", "ls": "-",  "lw": 2,
        },
    }
    for model in models:
        if "initial" in sim.fit[model].keys():
            kwargs_deB19[model]["label"] = kwargs_deB19[model]["label"].format(
                sim.fit[model]["deB19_chi2"])
            ax.plot(x0, sim.fit[model]["model"](x0, *sim.fit[model]["initial"]),
                **kwargs_deB19[model])
        if "soln" in sim.fit[model].keys():
            ax.plot(x0, sim.fit[model]["model"](x0, *sim.fit[model]["soln"].x),
                c="k", lw=2, alpha=0.7, label="{0} ML".format(model))

            ymodel = sim.fit[model]["model"](sim.fit_x, *sim.fit[model]["soln"].x)
            sigma2 = sim.fit_yerr**2
            chisq = numpy.sum((sim.fit_y - ymodel)**2 / sigma2)
            dof = len(sim.fit_x) - len(sim.fit[model]["soln"].x)
            print(chisq, len(sim.fit_x), chisq/dof)
        # if mcmc is not None:
        #     ax.plot(x0, model(x0, *mcmc, verbose=True),
        #         c="r", lw=2, label="Emcee")
        # if flat_samples is not None:
        #     inds = numpy.random.randint(len(flat_samples), size=size)
        #     for ind in inds:
        #         sample = flat_samples[ind]
        #         pyplot.plot(x0, model(x0, *sample[:3]), "C1", alpha=0.1)

    ax.legend(fontsize=20)
    return fig



def new_argument_parser():
    args = argparse.ArgumentParser(description="Fit Limepy Profiles to deBoer+ 2019 data")
    args.add_argument("-gc", "--gc_name", dest="gc_name", default="NGC 104",
        type=str, help="Name of the Globular Cluster")
    args.add_argument("-Nw", "--walkers", dest="Nwalkers", default=32,
        type=int, help="Number of MCMC walkers")
    args.add_argument("-Ns", "--samples", dest="Nsamples", default=5000,
        type=int, help="Number of MCMC samples")
    args.add_argument("-Nb", "--burn-in", dest="Nburn_in", default=500,
        type=int, help="Number of MCMC samples to discard to burn-in the chains")

    return args


if __name__ == "__main__":
    args, unknown = new_argument_parser().parse_known_args()

    import logging
    logging.getLogger("keyring").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__file__)
    logger.info("Running {0}".format(__file__))
    logger.info("  gc_name : {0}".format(args.gc_name))
    logger.info("  Nwalkers: {0}".format(args.Nwalkers))
    logger.info("  Nsamples: {0}".format(args.Nsamples))
    logger.info("  Nburn_in: {0}\n".format(args.Nburn_in))

    if "/supaharris" not in sys.path:
        sys.path.insert(0, "{}/supaharris".format(BASEDIR))
    from utils import parsec2arcmin

    if "../src" not in sys.path:
        sys.path.insert(0, "{}../src".format(BASEDIR))
    from gc_simulation import StarClusterSimulation

    # Run the MCMC fit
    sim = StarClusterSimulation(logger, args.gc_name)
    sim.fit_model_to_deBoer2019(mcmc=True, Nwalkers=args.Nwalkers, Nsamples=args.Nsamples)

    import pickle
    outdir = "{0}/out/".format("/".join(os.path.abspath(__file__).split("/")[:-2]))
    with open("{0}{1}_{2}_{3}_{4}.p".format(outdir, args.gc_name,
            args.Nwalkers, args.Nsamples, args.Nburn_in), "wb") as f:
        pickle.dump(sim, f)
    # with open("{0}{1}_{2}_{3}_{4}.p".format(outdir,  args.gc_name,
    #         args.Nwalkers, args.Nsamples, args.Nburn_in), "rb") as f:
    #     sim = pickle.load(f)

    # Remove more samples to burn in the walker
    sim.flat_samples = get_flat_samples(sim.sampler, sim.tau, discard=args.Nburn_in)

    pyplot.switch_backend("agg")
    pyplot.style.use("default")
    inspect_chains(sim.sampler, sim.fit_labels).savefig(
        "{0}mcmc_chains_{1}_{2}_{3}_{4}.png".format(outdir, args.gc_name,
            args.Nwalkers, args.Nsamples, args.Nburn_in)
    )
    plot_corner(sim.flat_samples, sim.initial, sim.fit_labels).savefig(
        "{0}mcmc_corner_{1}_{2}_{3}_{4}.png".format(outdir, args.gc_name,
            args.Nwalkers, args.Nsamples, args.Nburn_in)
    )
    pyplot.style.use("tlrh")
    plot_results(limepy_wrapper, sim, sim.fit_x, sim.fit_y, sim.fit_yerr,
        initial=sim.initial, ml=sim.soln.x, mcmc=sim.mcmc_mle,
        flat_samples=sim.flat_samples).savefig(
        "{0}mcmc_fit_{1}_{2}_{3}_{4}.png".format(outdir, args.gc_name,
            args.Nwalkers, args.Nsamples, args.Nburn_in)
    )
