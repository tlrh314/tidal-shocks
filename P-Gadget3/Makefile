#-----------------------------------------------------------------
#
# You might be looking for the compile-time Makefile options of the code...
#
# They have moved to a separate file.
#
# To build the code, do the following:
#
#  (1) Copy the file "Template-Config.sh"  to  "Config.sh"
#
#        cp Template-Config.sh Config.sh 
#
#  (2) Edit "Config.sh" as needed for your application
#
#  (3) Run "make"
#
#
#  New compile-time options should be added to the 
#  file "Template-Config.sh" only. Usually, the should be added
#  there in the disabled/default version.
#
#  "Config.sh" should *not* be checked in to the repository
#
#  Note: It is possible to override the default name of the 
#  Config.sh file, if desired, as well as the name of the
#  executable. For example:
#
#   make  CONFIG=MyNewConf.sh  EXEC=Gadget_new
# 
#-----------------------------------------------------------------
#
# You might also be looking for the target system SYSTYPE option
#
# It has also moved to a separate file.
#
# To build the code, do the following:
#
# (A) set the SYSTYPE variable in your .bashrc (or similar file):
#
#        e.g. export SYSTYPE=Magny
# or
#
# (B) set SYSTYPE in Makefile.systype 
#     This file has priority over your shell variable.:
#
#    (1) Copy the file "Template-Makefile.systype"  to  "Makefile.systype"
#
#        cp Template-Makefile.systype Makefile.systype 
#
#    (2) Uncomment your system in  "Makefile.systype".
#
# If you add an ifeq for a new system below, also add that systype to
# Template-Makefile.systype

ifdef SYSTYPE
SYSTYPE := "$(SYSTYPE)"
-include Makefile.systype
else
include Makefile.systype
endif

ifeq ($(wildcard Makefile.systype), Makefile.systype)
INCL = Makefile.systype
else
INCL =
endif


CONFIG   =  Config.sh
PERL     =  /usr/bin/perl

RESULT     := $(shell CONFIG=$(CONFIG) PERL=$(PERL) make -f config-makefile)
CONFIGVARS := $(shell cat gadgetconfig.h)


CC       = mpicc --cc=icc       # sets the C-compiler (default)
CXX       = mpiCC       # sets the C++-compiler (default)

FC 	 = mpif90

OPTIMIZE = -Wall  -g   # optimization and warning flags (default)

MPICHLIB = -lmpich

ifeq (SOFTDOUBLEDOUBLE,$(findstring SOFTDOUBLEDOUBLE,$(CONFIGVARS)))
CC       =   mpiCC     # default C++ compiler
OPTIMIZE =   -g 
OPT     +=  -DX86FIX   # only needed for 32-bit intel/amd systems
endif

ifeq (VORONOI,$(findstring VORONOI,$(CONFIGVARS)))
GMPLIB   =  -lgmp
endif

ifeq (NOTYPEPREFIX_FFTW,$(findstring NOTYPEPREFIX_FFTW,$(CONFIGVARS)))  # fftw installed without type prefix?
  FFTW_LIBNAMES =  -lrfftw_mpi -lfftw_mpi -lrfftw -lfftw
else
ifeq (DOUBLEPRECISION_FFTW,$(findstring DOUBLEPRECISION_FFTW,$(CONFIGVARS)))  # test for double precision libraries
  FFTW_LIBNAMES =  -ldrfftw_mpi -ldfftw_mpi -ldrfftw -ldfftw
else
  FFTW_LIBNAMES =  -lsrfftw_mpi -lsfftw_mpi -lsrfftw -lsfftw
endif
endif



#----------------------------------------------------------------------------------------------
# TODO: module load blabla
ifeq ($(SYSTYPE),"freya")
OPT     += -DNOCALLSOFSYSTEM
CC       = mpiicc 
OPTIMIZE = $(GADGETCFLAGS) -fopenmp -O3
GSL_INCL = -I$(GSL_HOME)/include
GSL_LIBS = -L$(GSL_HOME)/lib
FFTW_INCL= -I$(FFTW_HOME)/include
FFTW_LIBS= -L$(FFTW_HOME)/lib
MPICHLIB =
HDF5INCL = -I$(HDF5_HOME)/include
HDF5LIB  = -L$(HDF5_HOME)/bin -lhdf5 -lz -lsz
endif

# Debian 10 buster, /w kernel 4.9.0-6-amd64.
#   sudo apt-get install libopenmpi-dev libgsl-dev libfftw3-mpi-dev libhdf5-dev hdf5-tools
ifeq ($(SYSTYPE),"ZoltaN")
OPT          += -DNOCALLSOFSYSTEM
CC            = mpicc  # mpicc --version --> gcc (Debian 8.3.0-6) 8.3.0
OPTIMIZE      = $(GADGETCFLAGS) -fopenmp -O3
GSL_INCL      = -I/usr/include
GSL_LIBS      = -L/usr/lib/x86_64-linux-gnu  # gsl-config --version --> 2.5
FFTW_INCL     = -I/usr/include
FFTW_LIBS     = -L/usr/lib/x86_64-linux-gnu  # fftw-wisdom --version --> 3.3.8
FFTW_LIBNAMES =  -lfftw3_mpi -lfftw3_threads -lfftw3  # pkg-config --libs fftw3
MPICHLIB      =
HDF5INCL      = -I/usr/lib/x86_64-linux-gnu/hdf5/openmpi/include  # h5stat --version --> 1.10.4
HDF5LIB       = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib -lhdf5 -lz -lsz
endif

# Debian 10 buster, /w kernel 4.9.0-6-amd64.
#   sudo apt-get install libopenmpi-dev libgsl-dev libfftw3-mpi-dev libhdf5-dev hdf5-tools
ifeq ($(SYSTYPE),"Docker")
OPT          += -DNOCALLSOFSYSTEM
CC            = mpicc  # mpicc --version --> gcc (Debian 8.3.0-6) 8.3.0
OPTIMIZE      = $(GADGETCFLAGS) -fopenmp -O3
GSL_INCL      = -I/usr/include
GSL_LIBS      = -L/usr/lib/x86_64-linux-gnu  # gsl-config --version --> 2.5
FFTW_INCL     = -I/usr/include
FFTW_LIBS     = -L/usr/lib/x86_64-linux-gnu  # fftw-wisdom --version --> 3.3.8
FFTW_LIBNAMES =  -lfftw3_mpi -lfftw3_threads -lfftw3  # pkg-config --libs fftw3
MPICHLIB      =
HDF5INCL      = -I/usr/lib/x86_64-linux-gnu/hdf5/openmpi/include  # h5stat --version --> 1.10.4
HDF5LIB       = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib -lhdf5 -lz -lsz
endif

ifneq (HAVE_HDF5,$(findstring HAVE_HDF5,$(CONFIGVARS)))
ifneq (LT_METAL_COOLING_WAL,$(findstring LT_METAL_COOLING_WAL,$(CONFIGVARS)))
HDF5INCL =
HDF5LIB  =
endif
endif


ifeq (CHEMCOOL,$(findstring CHEMCOOL,$(CONFIGVARS)))
FOPT	+=  -WF,-DCHEMCOOL
OPT	+=  -DCHEMISTRYNETWORK=1
FOPT	+=  -WF,-DCHEMISTRYNETWORK=1
endif


ifneq (MODGRAV,$(findstring MODGRAV,$(CONFIGVARS))) # standard gravity object files
GRAVITY_OBJS  =  darkenergy.o pm_periodic.o pm_nonperiodic.o accel.o
else # replacements used in the modified gravity model 
GRAVITY_OBJS  =  modgrav/darkenergy.o modgrav/pm_periodic.o modgrav/pm_nonperiodic.o modgrav/accel.o
endif

L3_OBJS = 

ifeq (SORT_FROM_L3,$(findstring SORT_FROM_L3,$(CONFIGVARS))) # add L3 parallel sorting
L3_OBJS  +=  ImportFromL3/parallel_sort_l3.o
endif

ifeq (PM_FROM_L3,$(findstring PM_FROM_L3,$(CONFIGVARS))) # add L3 pm part
L3_OBJS  +=  ImportFromL3/pm_periodic_l3.o ImportFromL3/rfftwnd_mpi_threads.o ImportFromL3/transpose_mpi_threads.o
endif

ifeq (OPENACC_HYDRA,$(findstring OPENACC_HYDRA,$(CONFIGVARS))) # add OpenACC part                                                      
HYDRO_OBJS += OpenACC/hydra.o OpenACC/density.o
else
HYDRO_OBJS += hydra.o density.o
endif

# forcetree.o appears in two switches!
ifeq (OPENACC_GRAV,$(findstring OPENACC_GRAV,$(CONFIGVARS))) # OpenACC                                                      
GRAVITY_OBJS  +=    OpenACC/gravtree.o OpenACC/gpuallvars.o forcetree.o
else
GRAVITY_OBJS  +=    gravtree.o
ifeq (MODGRAV,$(findstring MODGRAV,$(CONFIGVARS))) # modgrav
GRAVITY_OBJS  +=    modgrav/forcetree.o 
else
GRAVITY_OBJS  +=    forcetree.o
endif
endif



OPTIONS = $(OPTIMIZE) $(OPT) 

FOPTIONS = $(OPTIMIZE) $(FOPT)

EXEC = P-Gadget3

OBJS  =  voronoi.o mpz_extension.o voronoi_density.o voronoi_force.o voronoi_2d.o voronoi_3d.o voronoi_exchange.o \
	 twopoint.o voronoi_meshrelax.o snIa_heating.o kspace_neutrinos.o kspace_neutrinos_2.o\
	 cosmic_rays_diffusion.o conduction.o fof.o subfind.o subfind_vars.o subfind_collective.o subfind_serial.o subfind_so.o subfind_cont.o \
	 subfind_distribute.o subfind_findlinkngb.o subfind_nearesttwo.o subfind_loctree.o subfind_alternative_collective.o subfind_reshuffle.o \
	 compare_partitions.o domain.o main.o greenf_diffusion.o driftfac.o growing_disk_potential.o \
	 subfind_potential.o subfind_density.o lineofsight.o kinfb_mhm.o sfr_mhm.o blackhole.o \
	 predict.o endrun.o global.o chemistry_noneq.o begrun.o run.o allvars.o\
	 timestep.o init.o restart.o io.o sfr_eff.o checksummed_sendrecv.o pinning.o \
	 forcetree_update.o read_ic.o read_ic_cluster.o cooling.o ngb.o parallel_sort.o second_order.o \
	 system.o allocate.o bubbles.o sizelimited_sendrecv.o rt_chem.o rt_bh_lum.o rt_sfr_lum.o rt_cooling.o \
	 kicks.o rt_eddington.o rt_n.o rt_CGmethod.o rt_stars_lum.o rt_gas_lum.o  \
	 potential.o phidot.o peano.o gravtree_forcetest.o parallel_sort_special.o \
	 longrange.o mpi_util.o mymalloc.o debug_md5/calc_checksum.o  debug_md5/Md5.o  \
	 cosmic_rays.o machfinder.o b_from_rot_a.o smooth_simple.o sub_turb_driving.o \
         phasespace.o phasespace_math.o helm_eos.o healpix.o compile_time_info.o \
         voronoi_ghost_search_alternative.o voronoi_ghost_search.o \
	 sinks.o \
	 cr_electrons.o \
	 scf.o scf_util.o \
	 mol_clouds.o 

OBJS	+= $(GRAVITY_OBJS)

OBJS	+= $(L3_OBJS)

OBJS	+= $(HYDRO_OBJS)

INCL    += cr_electrons.h allvars.h proto.h forcetree.h cooling.h domain.h  cosmic_rays.h chemistry.h \
	 machfinder.h subfind.h dd.h fof.h voronoi.h helm_eos.h myqsort.h chemcool_consts.h kernel.h Makefile \

ifeq (AR_GREEN_TREE,$(findstring AR_GREEN_TREE,$(CONFIGVARS)))
OBJS  += ar_green_tree/ar_ngb.o
endif

ifeq (SIDM,$(findstring SIDM,$(CONFIGVARS)))
OBJS    +=  sidm.o
INCL    +=
endif

ifeq (NUCLEAR_NETWORK,$(findstring NUCLEAR_NETWORK,$(CONFIGVARS)))
OBJS	+=  utilities.o integrate.o network_solver.o network.o 
INCL	+=  utilities.h integrate.h network_solver.h network.h
endif

ifeq (MODGRAV,$(findstring MODGRAV,$(CONFIGVARS)))
OBJS	+=  modgrav/modgrav_fieldsolve.o modgrav/modgrav_background.o modgrav/modgrav_cic.o modgrav/modgrav_effective_mass.o 
INCL	+=  modgrav/modgrav_cic.h modgrav/modgrav_effective_mass.h 
endif

ifeq (OUTPUT_LIGHTCONES,$(findstring OUTPUT_LIGHTCONES,$(CONFIGVARS)))
OBJS	+=  lightcones/lightcones.o
endif

ifeq (WINDTUNNEL,$(findstring WINDTUNNEL,$(CONFIGVARS)))
OBJS	+=  windtunnel/tunnel.o
INCL	+=  
endif

ifeq (CS_MODEL,$(findstring CS_MODEL,$(CONFIGVARS)))
OBJS  += cs_model/cs_metals.o cs_model/cs_sfr.o cs_model/cs_enrichment.o cs_model/cs_findhotngbs.o cs_model/cs_update_weights.o cs_model/cs_ngbfind.o
INCL  += cs_model/cs_metals.h
endif

ifeq (VS_TURB,$(findstring VS_TURB,$(CONFIGVARS)))
OBJS	+=  turb/turb_driving.o turb/turb_powerspectra.o
INCL	+=  
endif

ifeq (AB_TURB,$(findstring AB_TURB,$(CONFIGVARS)))
OBJS	+= turb/ab_turb.o  turb/turb_driving.o turb/turb_powerspectra.o
endif

ifeq (TURB_DRIVING,$(findstring TURB_DRIVING,$(CONFIGVARS)))
OBJS	+= fb_driveturb/turb_driving.o
endif

ifeq (LT_METAL_COOLING,$(findstring LT_METAL_COOLING,$(CONFIGVARS)))
OBJS  += lt_sfr/lt_wal_cooling.o lt_sfr/lt_sn.o lt_sfr/lt_io.o lt_sfr/lt_utils.o ImfLib/libimf.o ImfLib/libimf_vars.o 
INCL  += lt_sfr/lt_sn.h lt_sfr/lt.h lt_sfr/lt_error_codes.h lt_sfr/lt_wal_cooling.h 
ifeq (LT_SEv_INFO,$(findstring LT_SEv_INFO,$(CONFIGVARS)))
OBJS  += lt_sfr/lt_sfr.o
else
ifeq (LT_ZAGE,$(findstring LT_ZAGE,$(CONFIGVARS)))
OBJS  += lt_sfr/lt_sfr.o
else
OBJS  += lt_sfr/lt_sfr_light.o
endif
endif
endif

ifeq (LT_GM_FBMODEL,$(findstring LT_GM_FBMODEL,$(CONFIGVARS))) # add MUPPI 
OBJS += Muppi/sfr_muppi.o Muppi/muppi_communications.o
endif

ifeq (BP_REAL_CRs,$(findstring BP_REAL_CRs,$(CONFIGVARS))) # add bp cr part
OBJS += bp_cosmic_rays/bp_cosmic_rays.o
INCL += bp_cosmic_rays/bp_cosmic_rays.h
endif

ifeq (UM_CHEMISTRY,$(findstring UM_CHEMISTRY,$(CONFIGVARS)))
OBJS += um_chemistry/um_chemistry_noneq.o
endif

ifeq (UM_METAL_COOLING,$(findstring UM_METAL_COOLING,$(CONFIGVARS)))
OBJS += um_chemistry/um_metal_cooling.o 
endif

ifeq (ADAPTGRAVSOFT,$(findstring ADAPTGRAVSOFT,$(CONFIGVARS)))
INCL += AdaptGravSoft/ads_ngb.c AdaptGravSoft/ads_density.c
endif


ifeq (ADJ_BOX_POWERSPEC,$(findstring ADJ_BOX_POWERSPEC,$(CONFIGVARS)))
OBJS += power_spec/adj_box_powerspec.o 
INCL += power_spec/adj_box_powerspec_proto.h 
endif



ifeq (CHEMCOOL,$(findstring CHEMCOOL,$(CONFIGVARS)))

OBJS  += chemcool.o raytrace.o \

INCL  += chemcool_consts.h \

FOBJS = calc_photo.o calc_temp.o cheminmo.o compute_heating.o compute_md_kappa.o const_rates.o cool_func.o \
	cool_util.o coolinmo.o dvode.o evolve_abundances.o jac.o photoinit_lowZ.o \
	rate_eq_primordial.o spline.o validate_rates.o \

FINCL = cool.h fs_data.h mol_data.h non_eq.h shield_data.h Makefile \

else
FC    = $(CC)
endif

ifeq (EAGLE,$(findstring EAGLE,$(CONFIGVARS)))
OBJS  += eagle/eagle_cooling.o 
INCL  += eagle/eagle_cooling.h 
endif


CFLAGS = $(OPTIONS) $(GSL_INCL) $(FFTW_INCL) $(HDF5INCL) $(GMP_INCL)

ifeq (OUTPUT_LIGHTCONES,$(findstring OUTPUT_LIGHTCONES,$(CONFIGVARS)))
CFLAGS	+=  $(HEALPIX_INCL) 
endif

ifeq (VIP,$(findstring VIP,$(CONFIGVARS)))
FFLAGS = $(FOPTIONS)
else
FFLAGS = $(OPTIONS)
endif


ifeq (ALTERNATIVE_PSORT,$(findstring ALTERNATIVE_PSORT,$(CONFIGVARS)))
OBJS  += fof_alt_psort.o psort-1.0/error_handling.o
CXXFLAGS = $(CFLAGS)
FC    = $(CXX)
endif

FFTW = $(FFTW_LIBS)  $(FFTW_LIBNAMES) 


LIBS   = -lm $(HDF5LIB) -g $(MPICHLIB) $(GSL_LIBS) -lgsl -lgslcblas $(FFTW)

ifeq (NUM_THREADS,$(findstring NUM_THREADS,$(CONFIGVARS))) 
LIBS   +=  -lpthread
endif

ifeq (VORONOI,$(findstring VORONOI,$(CONFIGVARS)))
LIBS   +=  $(GMP_LIBS) $(GMPLIB)
endif

ifeq (GDE_BIGFLOAT,$(findstring GDE_BIGFLOAT,$(CONFIGVARS)))
LIBS   +=  -L/n/home12/mvogelsberger/opt/lib -lhpa
CFLAGS +=  -I/n/home12/mvogelsberger/opt/include 
endif

ifeq (OUTPUT_LIGHTCONES,$(findstring OUTPUT_LIGHTCONES,$(CONFIGVARS)))
LIBS   +=  $(HEALPIX_LIBS)
endif

$(EXEC): $(OBJS) $(FOBJS)  
	$(FC) $(OPTIMIZE) $(OBJS) $(FOBJS) $(LIBS) $(RLIBS) -o $(EXEC)

$(OBJS): $(INCL)  $(CONFIG)  compile_time_info.c


$(FOBJS): $(FINCL)

complile_time_info.c: $(CONFIG)
	$(PERL) prepare-config.perl $(CONFIG)

clean:
	rm -f $(OBJS) $(FOBJS) $(EXEC) gadgetconfig.h


help :
	@echo "SYSTYPE       = $(SYSTYPE)"
	@echo "OPT           = $(OPT)"
	@echo "CC            = $(CC)"
	@echo "OPTIMIZE      = $(OPTIMIZE)"
	@echo "GSL_INCL      = $(GSL_INCL)"
	@echo "GSL_LIBS      = $(GSL_LIBS)"
	@echo "FFTW_INCL     = $(FFTW_INCL)"
	@echo "FFTW_LIBS     = $(FFTW_LIB)"
	@echo "FFTW_LIBNAMES = $(FFTW_LIBNAMES)"
	@echo "MPICHLIB      = $(MPICHLIB)"
	@echo "HDF5INCL      = $(HDF5INCL)"
	@echo "HDF5LIB       = $(HDF5LIB)"
