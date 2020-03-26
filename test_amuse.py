from amuse.units import units
from amuse.units import generic_unit_converter
from amuse.ic.plummer import new_plummer_sphere
from amuse.community.gadget2.interface import Gadget2

converter = generic_unit_converter.ConvertBetweenGenericAndSiUnits(
    1 | units.MSun, 1 | units.parsec, 1 | units.Myr
)
gravity = Gadget2(
    channel_type="sockets",
    redirection="none",
    number_of_workers=12,
    unit_converter=converter,
)
gravity.parameters.time_max = 10 | units.Gyr

plummer = new_plummer_sphere(1000, convert_nbody=converter)
gravity.particles.add_particles(plummer)

gravity.evolve_model(1 | units.Gyr)
gravity.stop()
