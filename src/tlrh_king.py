import numpy

from amuse.units import units
from amuse.units import constants

from amuse.ic.kingmodel import MakeKingModel
# AMUSE's King (1966) model implementation, but w/o particles.
# Just want AMUSE to solve Poisson equation for King distribution function to give me
# rho(r), and MakeKingModel.poisson does just that and parks output in self.rr and self.d.
# The mass profile M(<r) is self.zm, the scaled potential is self.psi


## Obtained by numerically solving Poisson's equation
def king_density(W0):
    king = MakeKingModel(0, W0)  # King model /w 0 particles
    nprof, v20 = king.poisson()
    return king.rr, king.d


def king_mass(W0):
    king = MakeKingModel(0, W0)  # King model /w 0 particles
    nprof, v20 = king.poisson()
    return king.rr, king.zm


def king_potential(W0):
    king = MakeKingModel(0, W0)  # King model /w 0 particles
    nprof, v20 = king.poisson()
    return king.rr, king.psi


def king_velocity_dispersion(r, M, a):
    king = MakeKingModel(0, W0)  # King model /w 0 particles
    nprof, v20 = king.poisson()
    return king.rr, king.v2


def king_distribution_function(p, r):
    raise NotImplementedError
    return


def king_core_radius(a):
    return


def king_half_mass_radius(a):
    return


def king_virial_radius(a):
    return


## Diagnostic / plots
def add_king_density_profile_to_ax(ax, p, a, r_ana):
    Mtot = p.total_mass().as_quantity_in(units.MSun)
    rho = king_density(r_ana, Mtot, a)
    ax.plot(
        r_ana.value_in(units.parsec),
        rho.value_in(units.MSun * units.parsec**(-3)),
        c="k", ls="--", label="king (analytical)"
    )


def add_king_mass_profile_to_ax(ax, p, a):
    Mtot = p.total_mass().as_quantity_in(units.MSun)
    mass = king_mass(r_ana, Mtot, a)
    ax.plot(
        r_ana.value_in(units.parsec),
        mass.value_in(units.MSun),
        c="k", ls="--", label="analytical"
    )

def add_king_radii_to_ax(ax, p, a, debug=False):
    # Indicate king radius, Core radius, Half-mass radius, Virial radius

    # TODO: get half-mass and virial radius numerically (from datamodel.Particles)?

    rc = king_core_radius(a)
    rh = king_half_mass_radius(a)
    rv = king_virial_radius(a)

    if debug:
        print("Adding king radii to ax")
        print("  a = {0}".format(a))
        print("  rc = {0}".format(rc))
        print("  rh = {0}".format(rh))
        print("  rv = {0}".format(rv))

    trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for rname, r in zip([r"$r_{\rm king}$", r"$r_{\rm core}$",
            r"$r_{\rm half-mass}$", r"$r_{\rm virial}$"],
            [a, rc, rh, rv]):
        ax.axvline(r.value_in(units.parsec), c="k", ls=":")
        ax.text(r.value_in(units.parsec), 1.02,
            r"{0} = {1:.2f}".format(rname, r.value_in(units.parsec)),
            rotation=45, color="k", fontsize=14,
            ha="left", va="bottom", transform=trans
        )


if __name__ == "__main__":
    Nstars = 50000
    W0 = 5

    from amuse.units import nbody_system
    convert_nbody = nbody_system.nbody_to_si(Mtotal, Rtidal)
    print("converter.length = {0} parsec\nconverter.mass = {1} MSun\n"
          "converter.time = {2} yr".format(
        (convert_nbody.units[0][1]).value_in(units.parsec),
        (convert_nbody.units[1][1]).value_in(units.MSun),
        (convert_nbody.units[2][1]).value_in(units.yr)
    ))

    from amuse.ic.kingmodel import new_king_model
    king = new_king_model(Nstars, W0, nbody_converter)
