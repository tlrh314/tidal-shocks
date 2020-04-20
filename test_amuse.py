print("test_amuse.py line 1")
from amuse.units import units
print("amuse.units.units imported")
from amuse.units import generic_unit_converter
print("amuse.units.generic_unit_converter imported")
from amuse.ic.plummer import new_plummer_sphere
print("amuse.ic.plummer.new_plummer_sphere imported")
from amuse.community.gadget2.interface import Gadget2
print("amuse.community.gadget2.interface imported")

converter = generic_unit_converter.ConvertBetweenGenericAndSiUnits(
    1 | units.MSun, 1 | units.parsec, 1 | units.Myr
)
print("converter initialised")
gravity = Gadget2(
    debugger="strace",
    redirection="none",
    number_of_workers=4,
    unit_converter=converter,
)
print("Gadget2 initialised")
plummer = new_plummer_sphere(1000, convert_nbody=converter)
print("new_plummer_sphere initialised")
gravity.particles.add_particles(plummer)
print("particles added to Gadget2")

gravity.evolve_model(1 | units.Myr)
print("evolve_model done")
gravity.stop()
print("Gadget2 stopped")
