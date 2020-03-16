import re
import sys
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


def limepy_wrapper(x, W0, M, rt, g=1, verbose=False):
    if verbose:
        print("Limepy, W0={0:.3f}, M={1:.3e}, rt={2:.3f}, g={3:.3f}".format(
            W0, M, rt, g))
    lime = limepy.limepy(W0, M=M, rt=rt, g=1, project=True, verbose=False)

    # Interpolate the projected surface density profile to allow for arbitrary radii
    interp1d = scipy.interpolate.interp1d(lime.R, lime.Sigma)

    # However, interp1d needs to stay within bounds. The Limepy profile is
    # only sampled up to the truncation radius rt, and cannot be interpolated
    # beyond. The surface brightness profile, by construction, is zero at the
    # truncation radius. So we set the counts to ZERO equal to small value
    ZERO = 1e-9
    counts = numpy.array([interp1d(xi) if xi < lime.rt else ZERO for xi in x])

    return counts


def log_likelihood(theta, x, y, yerr, model):
    W0, M, rt = theta
    ymodel = model(x, W0, M, rt)
    sigma2 = yerr ** 2
    # Note that I added 2*numpy.pi in the logarithm of sigma2 wrt documentation
    return -0.5 * numpy.sum((y - ymodel) ** 2 / sigma2 + numpy.log(2*numpy.pi*sigma2))


def log_prior(theta):
    W0, M, rt = theta
    # TODO: why 0.0 if parameter in range, and -inf if parameter not in range?
    # The documentation shows that p(m) is 1/5.5 if -5 < m < 1/2, but this
    # method returns 0.0. That does not make much sense, does it?
    # TODO: may want lognormal weak priors rather than this business of
    # "uninformative flat priors"
    if 0 < W0 < 14 and 1e2 < M < 1e6 and 1 < rt < 300:
        return 0.0
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


def plot_results(model, sim, x, y, yerr, initial=None, ml=None, mcmc=None,
        flat_samples=None, size=100, x0=numpy.logspace(-3, 3, 256)):
    fig, ax = pyplot.subplots(1, 1, figsize=(12, 9))
    sim.add_deBoer2019_to_fig(fig)
    ax.errorbar(x, y, yerr=yerr, fmt="ro", ms=4, capsize=0, label="Subset")
    if initial is not None:
        ax.plot(x0, model(x0, *initial, verbose=True),
            c="g", lw=2, label="initial (deB19)")
    if ml is not None:
        ax.plot(x0, model(x0, *ml, verbose=True),
            c="k", lw=2, label="ML")
    if mcmc is not None:
        ax.plot(x0, model(x0, *mcmc, verbose=True),
            c="r", lw=2, label="Emcee")
    if flat_samples is not None:
        inds = numpy.random.randint(len(flat_samples), size=size)
        for ind in inds:
            sample = flat_samples[ind]
            pyplot.plot(x0, model(x0, *sample[:3]), "C1", alpha=0.1)

    ax.legend(fontsize=20)
    return fig


if __name__ == "__main__":
    import logging
    logging.getLogger("keyring").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__file__)
    logger.info("Running {0}".format(__file__))

    if "/supaharris" not in sys.path:
        sys.path.insert(0, "{}/supaharris".format(BASEDIR))
    from utils import parsec2arcmin

    if "../src" not in sys.path:
        sys.path.insert(0, "{}../src".format(BASEDIR))
    from gc_simulation import StarClusterSimulation

    # Run the MCMC fit
    gc_name = "NGC 104"
    sim = StarClusterSimulation(logger, gc_name)
    sim.fit_model_to_deBoer2019(mcmc=True, Nwalkers=8, Nsamples=100)

    # Remove more samples to burn in the walker
    # sim.flat_samples = get_flat_samples(sim.sampler, sim.tau, discard=500)

    outdir = "{0}/out/".format("/".join(os.path.abspath(__file__).split("/")[:-2]))
    pyplot.style.use("default")
    inspect_chains(sim.sampler, sim.fit_labels).savefig(
        "{0}mcmc_chains_{1}.png".format(outdir, gc_name)
    )
    plot_corner(sim.flat_samples, sim.initial, sim.fit_labels).savefig(
        "{0}mcmc_corner_{1}.png".format(outdir, gc_name)
    )
    pyplot.style.use("tlrh")
    plot_results(limepy_wrapper, sim, sim.fit_x, sim.fit_y, sim.fit_yerr,
        initial=sim.initial, ml=sim.soln.x, mcmc=sim.mcmc_mle,
        flat_samples=sim.flat_samples).savefig(
        "{0}mcmc_fit_{1}.png".format(outdir, gc_name)
    )
