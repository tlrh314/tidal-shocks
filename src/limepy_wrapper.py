import sys
import time
import platform
import numpy
import scipy


BASEDIR = "/u/timoh/phd/" if "freya" in platform.node() else ""
if "/limepy" not in sys.path:
    sys.path.insert(0, "{}/limepy".format(BASEDIR))
import limepy   # using tlrh314/limepy fork

ZERO = 1e-9


def king_wrapper(x, W0, M, rt, verbose=False):
    return limepy_wrapper(x, W0, M, rt, 1, verbose=verbose)


def wilson_wrapper(x, W0, M, rt, verbose=False):
    return limepy_wrapper(x, W0, M, rt, 2, verbose=verbose)


def limepy_wrapper(x, W0, M, rt, g, verbose=False):
    if verbose:
        print("  Limepy, W0={0:.3f}, M={1:.3e}, rt={2:.3f}, g={3:.3f}".format(
            W0, M, rt, g))
    lime = limepy.limepy(W0, M=M, rt=rt, g=g, project=True, verbose=False)

    # Interpolate the projected surface density profile to allow for arbitrary radii
    interp1d = scipy.interpolate.interp1d(lime.R, lime.Sigma)

    # However, interp1d needs to stay within bounds. The Limepy profile is
    # only sampled up to the truncation radius rt, and cannot be interpolated
    # beyond. The surface brightness profile, by construction, is zero at the
    # truncation radius. So we set the counts to ZERO equal to small value
    counts = numpy.array([interp1d(xi) if xi < lime.rt else ZERO for xi in x])

    return counts


def spes_wrapper(x, W0, B, eta, M, rt, nrt, verbose=False):
    # Prevent infinite loop when spes has B or eta out of bounds
    if ( (0.0 > W0 or W0 > 14) or (0.0 > B or B > 1.0) or (0.0 > eta or eta > 1.0) or
         (1e2 > M or M > 1e6) or (1.0 > rt or rt > 300) or (0.1 > nrt or nrt > 100) ):
        return numpy.ones(len(x))

    if verbose:
        print("  Spes, W0={0:.3f}, B={1:.3f}, eta={2:.3f}, M={3:.3e}, rt={4:.3f}, nrt={5:.3f}".format(
            W0, B, eta, M, rt, nrt))
    start = time.time()
    spes = limepy.spes(W0, B=B, eta=eta, M=M, rt=rt, nrt=nrt, project=True, verbose=False)

    # Interpolate the projected surface density profile to allow for arbitrary radii
    interp1d = scipy.interpolate.interp1d(spes.R, spes.Sigma)

    # However, interp1d needs to stay within bounds. The Limepy profile is
    # only sampled up to the truncation radius rt, and cannot be interpolated
    # beyond. The surface brightness profile, by construction, is zero at the
    # truncation radius. So we set the counts to ZERO equal to small value
    counts = numpy.array([interp1d(xi) if xi < spes.rt else ZERO for xi in x])

    return counts
