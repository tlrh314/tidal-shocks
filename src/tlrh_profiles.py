import numpy
import scipy
import limepy
from scipy import stats
from matplotlib import pyplot

# Define the statistical model, in this case we shall use a chi-squared
# distribution, assuming normality in the errors
def chisq(x, y, dy, ymodel):
    """ Statistical model. Here: Pearson's chi^2 assuming normality in errors
        @param x: independent variable, here the projected radius R [arcmin]
        @param y: dependent variable, here the star counts [arcmin^-2]
        @param dy: uncertainty in the dependent variable
        @param ymodel: function that returns ymodel(x)
        @return chi^2, given the model parameters """

    chi2 = numpy.sum( (y - ymodel)**2 / dy**2 )
    return chi2


def fit(logger, deB19_fits, deB19_stitched, gc_name, g=1, verbose=True):
    logger.debug("\n{0}\n{1}\n".format(gc_name, deB19_stitched[gc_name]))
    x =  deB19_stitched[gc_name]["rad"]
    y =  deB19_stitched[gc_name]["density"]
    dy = deB19_stitched[gc_name]["density_err"]

    # Because we need the distance
    from data.parse_harris_1996ed2010 import parse_harris1996ed2010
    h96_gc = parse_harris1996ed2010(logger)
    distance_kpc = h96_gc[gc_name].dist_from_sun

    def king(x, W0, M, rt):
        print("King, W0={0:.1f}, M={1:.1e}, rt={2:.1f}, {3}".format(W0, M, rt, COLOR))

        # Set up the King (1966) model /w Limepy
        king = limepy.limepy(W0, M=M, rt=rt, g=1, project=True, verbose=False)

        # Interpolate the projected surface density profile to allow for arbitrary radii
        interp1d = scipy.interpolate.interp1d(king.R, king.Sigma)

        # However, interp1d needs to stay within bounds. The Limepy profile is
        # only sampled up to the truncation radius rt, and cannot be interpolated
        # beyond. The surface brightness profile, by construction, is zero at the
        # truncation radius. So we set the counts to ZERO equal to small value
        ZERO = 1e-9
        counts = [interp1d(xi) if xi < king.rt else ZERO for xi in x]

        fig, ax = pyplot.gcf(), pyplot.gca()
        # ax.plot(king.R, king.Sigma, label="W0={0:.1f}, M={1:.1e}, rt={2:.1f}".format(W0, M, rt))
        ax.plot(king.R, king.Sigma, c=COLOR, lw=1, ls=":", alpha=0.5)
        return counts
    ymodel = lambda x, W0, M, rt: king(x, W0, M, rt)
    chisq_king = lambda p, x, y, dy: chisq(x, y, dy, ymodel(x, p[0], p[1], p[2]))

    # Setup the initial guess of the MLEs
    igc, = numpy.where(deB19_fits["id"] == gc_name)[0]
    irt_guess, = numpy.argwhere(y <= deB19_fits[igc]["BGlev"])[0]
    rt_guess = x[irt_guess]  # in arcmin

    x0 = [6.8, 2753, 8]
    x0 = [5, max(y), rt_guess]
    bounds = [(1, 14), (1e2, 1e6), (1, 50)]
    # bounds = [(None, None), (None, None), (None, None)]

    pyplot.switch_backend("agg")
    fig, ax = pyplot.subplots(1,1, figsize=(12,9))
    fig = plot_deBoer_2019(logger, deB19_fits[igc], deB19_stitched, fig=fig,
        show_King=False, show_Wilson=False, show_limepy=False, show_spes=False,
        show_BGlev=True, show_rtie=True, show_rJ=True, has_tex=False, verbose=False)

    # Minimise chi^2 to obtain best-fit parameters
    # minimize(fun, x0, args=()) where fun(x, *args) --> float
    COLOR = "green"
    result = scipy.optimize.minimize(chisq_king, x0, args=(x, y, dy))
    #     method='L-BFGS-B', bounds=bounds)  # b/c L-BFGS-B allows bounds
    print(result)

    ml_vals = result["x"]
    ml_func = result["fun"]

    x_ana = numpy.logspace(-3, 3, 64)
    ax.plot(x_ana, king(x_ana, x0[0], x0[1], x0[2]), label="x0: {0}".format(x0))
    ax.plot(x_ana, king(x_ana, ml_vals[0], ml_vals[1], ml_vals[2]), label="ml: {0}".format(ml_vals))

    # Obtain degrees of freedom and check goodness-of-fit. This is useless tho
    moddof = len(ml_vals)  # Model degrees of freedom; nr of fit parameters
    # Here we count the unmasked values. NB id(i) != id(True)
    dof = len(x) - moddof  # degrees of freedom
    ch = scipy.stats.chi2(dof)
    pval = 1.0 - ch.cdf(ml_func)

    # Obtain MLEs using Scipy's curve_fit which gives covariance matrix
    COLOR = "red"
    ml_vals, ml_covar = scipy.optimize.curve_fit(ymodel, x, y,
        p0=ml_vals, sigma=dy, method="trf", bounds=[b for b in zip(*bounds)])

    print(result["x"], "(minimise chisq)")
    print(ml_vals, "(curve_fit)")

    if not result["success"]:
        print("  scipy.optimize.curve_fit broke down!\n    Reason: '{0}'"\
            .format(result["message"]))
        print("  No confidence intervals have been calculated.")
        print(ml_vals, "\n", ml_covar)
        # import sys; sys.exit(1)

    # Calculate the uncertainties on the MLEs from the covariance matrix
    err = numpy.sqrt(numpy.diag(ml_covar))

    if verbose:
        print("Results for the King model:")
        print("  Using scipy.optimize.minimize to minimize chi^2 yields:")
        print("    W0          = {0:.5f}".format(result["x"][0]))
        print("    M           = {0:.5f}".format(result["x"][1]))
        print("    r_t         = {0:.5f}".format(result["x"][2]))
        print("    chisq       = {0:.5f}".format(ml_func))
        print("    dof         = {0:.5f}".format(dof))
        print("    chisq/dof   = {0:.5f}".format(ml_func/dof))
        print("    p-value     = {0:.5f}".format(pval))
        print("  Using scipy.optimize.curve_fit to obtain confidence intervals yields:")
        print("    W0          = {0:.5f} +/- {1:.5f}".format(ml_vals[0], err[0]))
        print("    M           = {0:.5f} +/- {1:.5f}".format(ml_vals[1], err[1]))
        print("    r_t         = {0:.4f} +/- {1:.4f}".format(ml_vals[2], err[2]))

    ax.plot(x_ana, king(x_ana, ml_vals[0], ml_vals[1], ml_vals[2]), label="curve_fit: {0}".format(ml_vals))
    ax.legend(loc="best", fontsize=16, frameon=True)
    fig.savefig("/tidalshocks/out/{0}.pdf".format(gc_name))

    return ml_vals, err


def fit_king(p, x, y, dy, verbose=True):
    """ Fit King (1966) model /w parameters p (tuple) to data (x, y, dy)
        @param p: central value of potential W0, total mass M, truncation radius rt
    """

    ymodel = limepy.limepy(p[0], M=p[1], rt=p[2], g=1, project=True, verbose=verbose)


def fit_wilson(p, x, y, dy):
    pass


def fit_limepy(p, x, y, dy):
    pass


def fit_spes(p, x, y, dy):
    pass


if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__file__)
    logger.info("Running {0}".format(__file__))

    # Get deBoer+ 2019 data to fit to real data
    sys.path.insert(0, "/supaharris")
    from data.parse_deBoer_2019 import plot_deBoer_2019
    from data.parse_deBoer_2019 import parse_deBoer_2019_fits
    from data.parse_deBoer_2019 import parse_deBoer_2019_stitched_profiles
    deB19_fits = parse_deBoer_2019_fits(logger)
    deB19_stitched = parse_deBoer_2019_stitched_profiles(logger)
    logger.debug("\ndeBoer+ 2019")
    logger.debug("  Found {0} GCs".format(len(deB19_fits)))
    logger.debug("  Available fields:\n    {0}".format(deB19_fits.dtype))
    logger.debug("  Available clusters:\n    {0}".format(deB19_fits["id"]))

    # Fit King, Wilson, Limepy, SPES models to NGC 1261
    ml_vals, err = fit(logger, deB19_fits, deB19_stitched, "NGC 1261", g=1)  # g=1 --> King
