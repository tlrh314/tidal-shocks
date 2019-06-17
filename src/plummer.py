import numpy
import matplotlib

from amuse.units import units
from amuse.units import constants


## Analytical
def plummer_density(r, M, a):
    return (3*M) / (4 * numpy.pi * a**3) * (1 + r**2/a**2)**(-5.0/2)


def plummer_mass(r, M, a):
    return M * ( r**3 / (r**2 + a**2)**(3.0/2) )


def plummer_potential(r, M, a):
    return -1.0*constants.G*M / (r**2 + a**2).sqrt()


def plummer_velocity_dispersion(r, M, a):
    return constants.G*M / (6 * (r**2 + a**2).sqrt())


def plummer_distribution_function(p, r):
    N = len(stars)
    M = p.total_mass().as_quantity_in(units.MSun)
    return (24 * numpy.sqrt(2)) / (7*numpy.pi**3) * (N*a**2) / (contstants.G**5 * M**5) * E(p.x, p.v)


def plummer_core_radius(a):
    return a * numpy.sqrt( numpy.sqrt(2) - 1 )  # core radius ~ 0.64*a


def plummer_half_mass_radius(a):
    return a * numpy.sqrt( (1/0.5**(2.0/3)) - 1)  # half-mass radius ~ 1.3*a


def plummer_virial_radius(a):
    return 16.0/(3*numpy.pi) * a  # virial radius ~ 1.7*a


## Diagnostic / plots
def add_plummer_density_profile_to_ax(ax, p, a, r_ana):
    Mtot = p.total_mass().as_quantity_in(units.MSun)
    rho = plummer_density(r_ana, Mtot, a)
    ax.plot(
        r_ana.value_in(units.parsec),
        rho.value_in(units.MSun * units.parsec**(-3)),
        c="k", ls="--", label="Plummer (analytical)"
    )


def add_plummer_mass_profile_to_ax(ax, p, a, r_ana):
    Mtot = p.total_mass().as_quantity_in(units.MSun)
    mass = plummer_mass(r_ana, Mtot, a)
    ax.plot(
        r_ana.value_in(units.parsec),
        mass.value_in(units.MSun),
        c="k", ls="--", label="analytical"
    )

def add_plummer_radii_to_ax(ax, p, a, debug=False):
    # Indicate Plummer radius, Core radius, Half-mass radius, Virial radius

    # TODO: get half-mass and virial radius numerically (from datamodel.Particles)?

    rc = plummer_core_radius(a)
    rh = plummer_half_mass_radius(a)
    rv = plummer_virial_radius(a)

    if debug:
        print("Adding Plummer radii to ax")
        print("  a = {0}".format(a))
        print("  rc = {0}".format(rc))
        print("  rh = {0}".format(rh))
        print("  rv = {0}".format(rv))

    trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for rname, r in zip([r"$r_{\rm plummer}$", r"$r_{\rm core}$",
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
    Mtotal = 30000 | units.MSun
    Rtidal = 1 | units.parsec

    from amuse.units import nbody_system
    convert_nbody = nbody_system.nbody_to_si(Mtotal, Rtidal)
    print("converter.length = {0} parsec\nconverter.mass = {1} MSun\n"
          "converter.time = {2} yr".format(
        (convert_nbody.units[0][1]).value_in(units.parsec),
        (convert_nbody.units[1][1]).value_in(units.MSun),
        (convert_nbody.units[2][1]).value_in(units.yr)
    ))

    from amuse.ic.plummer import new_plummer_sphere
    plummer = new_plummer_sphere(Nstars, convert_nbody)
