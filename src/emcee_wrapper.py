import os
import sys
import numpy
import scipy
import argparse
from matplotlib import pyplot
pyplot.style.use("tlrh")

import emcee
import corner
from multiprocessing import Pool
from emcee.autocorr import AutocorrError


def log_likelihood(theta, x, y, yerr, function):
    ymodel = function(x, *theta)
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
    elif len(theta) == 6:  # SPES
        W0, B, eta, M, rt, nrt = theta
        if ( 0 < W0 < 14 and 0 < B < 1 and 0 < eta < 1 and 1e2 < M < 1e6
            and 1 < rt < 300 and 1 < nrt < 100 ):
            return 0.0
    # TODO: why 0.0 if parameter in range, and -inf if parameter not in range?
    # The documentation shows that p(m) is 1/5.5 if -5 < m < 1/2, but this
    # method returns 0.0. That does not make much sense, does it?
    # TODO: may want lognormal weak priors rather than this business of
    # "uninformative flat priors"
    return -numpy.inf


def log_probability(theta, x, y, yerr, function):
    lp = log_prior(theta)
    if not numpy.isfinite(lp):
        return -numpy.inf
    return lp + log_likelihood(theta, x, y, yerr, function)


def minimise_chisq(initial, x, y, yerr, function):
    negative_log_likelihood = lambda *args: -log_likelihood(*args)
    return scipy.optimize.minimize(negative_log_likelihood, initial,
        args=(x, y, yerr, function))


def run_mcmc(ml, x, y, yerr, function, outdir, model_name, Nwalkers=32,
        Nsamples=500, rerun=False, progress=True):
    Ndim = len(ml)
    pos = ml + 1e-4 * numpy.random.randn(Nwalkers, Ndim)

    fname = "{}Emcee_{}_{}_{}.h5".format(outdir, model_name, Nwalkers, Nsamples)
    backend = emcee.backends.HDFBackend(fname)
    if os.path.exists(fname) and os.path.isfile(fname) and not rerun:
        print("  run_mcmc --> file exists: {}".format(fname))
        return backend

    backend.reset(Nwalkers, Ndim)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(Nwalkers, Ndim, log_probability,
            args=(x, y, yerr, function), backend=backend, pool=pool)
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


def get_tau(sampler, verbose=False):
    # Using tol=0 means that we'll always get an estimate even if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    if verbose: print("  tau: {0}".format(tau))
    return tau


def get_flat_samples(sampler, tau, discard=None, thin=None, verbose=False):
    mean_tau = int(numpy.ceil(numpy.mean(tau)))
    if discard is None:
        discard = int(2* numpy.max(tau))
    if thin is None:
        thin = int(0.5 * numpy.min(tau))
    samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    log_prob_samples = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
    log_prior_samples = sampler.get_blobs(discard=discard, thin=thin, flat=True)

    if verbose:
        print("  burn-in: {0}".format(discard))
        print("  thin: {0}".format(thin))
        print("  flat chain shape: {0}".format(samples.shape))
        print("  flat log prob shape: {0}".format(log_prob_samples.shape))
        print("  flat log prior shape: {0}".format(log_prior_samples.shape))

    return samples


def plot_corner(flat_samples, truths, labels):
    # Use the best-fit MLEs from deBoer+ 2019 as 'truths'
    ndim = len(truths)
    fig, axes = pyplot.subplots(ndim, ndim, figsize=(16, 16), facecolor="white")
    return corner.corner(flat_samples, bins=64, labels=labels, show_titles=True,
        truths=truths, plot_contours=False, quantiles=[0.16, 0.50, 0.84],
        top_ticks=True, fig=fig
    )


def plot_results(obs, model_names=["king", "wilson", "limepy", "spes"],
        x0=numpy.logspace(-3, 3, 256), size=100):
    fig, ax = pyplot.subplots(1, 1, figsize=(12, 12))
    obs.add_deBoer2019_to_fig(fig)
    ax.errorbar(obs.fit_x, obs.fit_y, yerr=obs.fit_yerr, fmt="ro",
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
    for model_name in model_names:
        if "initial" in obs.fit[model_name].keys():
            kwargs_deB19[model_name]["label"] = kwargs_deB19[model_name]["label"].format(
                obs.fit[model_name]["deB19_chi2"])
            ax.plot(x0, obs.fit[model_name]["function"](x0, *obs.fit[model_name]["initial"]),
                **kwargs_deB19[model_name])
        if "soln" in obs.fit[model_name].keys():
            ax.plot(x0, obs.fit[model_name]["function"](x0, *obs.fit[model_name]["soln"].x),
                c="k", lw=2, alpha=0.7, label="{0} ML, 'chi2'={1:.1f}".format(
                    model_name, obs.fit[model_name]["soln"].fun))

            # TODO: soln.fun and calculated chi2 inconsistent because we actually
            # minimize -log_likelihood, which seems to be -0.5chi^2 + ln(2*pi*sigma2).
            # However, neither calculation retrieves deB19_chi2 :-(
            ymodel = obs.fit[model_name]["function"](
                obs.fit_x, *obs.fit[model_name]["soln"].x)
            sigma2 = obs.fit_yerr**2
            chisq = numpy.sum((obs.fit_y - ymodel)**2 / sigma2)
            dof = len(obs.fit_x) - len(obs.fit[model_name]["soln"].x)
            print(chisq, 0.5*chisq, len(obs.fit_x), chisq/dof)
        if "mcmc_mle" in obs.fit[model_name].keys():
            ax.plot(x0, obs.fit[model_name]["function"](x0,
                *obs.fit[model_name]["mcmc_mle"], verbose=True),
                c="r", lw=2, label="Emcee")
        if "flat_samples" in obs.fit[model_name].keys():
            size = min( size, len(obs.fit[model_name]["flat_samples"]) )
            inds = numpy.random.randint(len(obs.fit[model_name]["flat_samples"]), size=size)
            for ind in inds:
                sample = obs.fit[model_name]["flat_samples"][ind]
                pyplot.plot(x0, obs.fit[model_name]["function"](
                    x0, *sample), "C1", alpha=0.1)

    ax.legend(fontsize=20)
    return fig


def new_argument_parser():
    args = argparse.ArgumentParser(
        description="Fit Limepy Profiles to deBoer+ 2019 data."
    )

    args.add_argument("-gc", "--gc_name", dest="gc_name", default="NGC 104",
        type=str, help="Name of the Globular Cluster."
    )
    args.add_argument("-m", "--model_name", dest="model_name", default="king",
        type=str, choices=["king", "wilson", "limepy", "spes"],
        help="Physical model for the density structure of the Globular Cluster.",
    )
    args.add_argument("-Nw", "--walkers", dest="Nwalkers", default=32,
        type=int, help="Number of MCMC walkers."
    )
    args.add_argument("-Ns", "--samples", dest="Nsamples", default=5000,
        type=int, help="Number of MCMC samples."
    )
    args.add_argument("-Nb", "--burn-in", dest="Nburn_in", default=500,
        type=int, help="Number of MCMC samples to discard to burn-in the chains."
    )
    args.add_argument("--seed", dest="seed", default=-1, type=int,
        help="Random number seed (-1, no seed).",
    )

    return args


if __name__ == "__main__":
    import logging
    logging.getLogger("keyring").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__file__)

    args, unknown = new_argument_parser().parse_known_args()

    logger.info("Running {}".format(__file__))
    logger.info("  gc_name: {}".format(args.gc_name))
    logger.info("  model_name: {}".format(args.model_name))
    logger.info("  Nwalkers: {}".format(args.Nwalkers))
    logger.info("  Nsamples: {}".format(args.Nsamples))
    logger.info("  Nburn_in: {}".format(args.Nburn_in))
    logger.info("  seed: {}\n".format(args.seed))

    if args.seed >= 0:
        numpy.random.seed(args.seed)

    # Some builds of NumPy (including the version included with Anaconda) will
    # automatically parallelize some operations using something like the MKL
    # linear algebra. This can cause problems when used with the parallelization
    # methods described here so it can be good to turn that off (by setting the
    # environment variable OMP_NUM_THREADS=1, for example).
    os.environ["OMP_NUM_THREADS"] = "1"

    # Run the MCMC fit
    from mw_gc_observation import MwGcObservation
    obs = MwGcObservation(logger, args.gc_name)
    obs.fit_model_to_deBoer2019(model_name=args.model_name, mcmc=True,
        Nwalkers=args.Nwalkers, Nsamples=args.Nsamples)

    # Remove more samples to burn in the walker
    obs.fit[args.model_name]["flat_samples"] = get_flat_samples(
        obs.fit[args.model_name]["sampler"], obs.fit[args.model_name]["tau"],
        discard=args.Nburn_in
    )

    pyplot.switch_backend("agg")
    pyplot.style.use("default")
    inspect_chains(obs.fit[args.model_name]["sampler"],
        obs.fit[args.model_name]["fit_labels"]).savefig(
        "{0}mcmc_chains_{1}_{2}_{3}_{4}.png".format(obs.outdir, obs.gc_slug,
            args.model_name, args.Nwalkers, args.Nsamples, args.Nburn_in)
    )
    plot_corner(obs.fit[args.model_name]["flat_samples"],
        obs.fit[args.model_name]["initial"], obs.fit[args.model_name]["fit_labels"]
    ).savefig("{0}mcmc_corner_{1}_{2}_{3}_{4}_{5}.png".format(obs.outdir, obs.gc_slug,
        args.model_name, args.Nwalkers, args.Nsamples, args.Nburn_in)
    )
    pyplot.style.use("tlrh")
    plot_results(obs, model_names=[args.model_name]).savefig(
        "{0}mcmc_fit_{1}_{2}_{3}_{4}_{5}.png".format(obs.outdir, obs.gc_slug,
            args.model_name, args.Nwalkers, args.Nsamples, args.Nburn_in)
    )
