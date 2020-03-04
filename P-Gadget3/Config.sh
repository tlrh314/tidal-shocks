#!/bin/bash            # this line only there to enable syntax highlighting in this file

# PERIODIC
# GRAVITY_NOT_PERIODIC

# OUTPUT_DIV_CURL
# OUTPUTPOTENTIAL

UNEQUALSOFTENINGS
MYSORT
MULTIPLEDOMAINS=16
TOPNODEFACTOR=16
OVERRIDE_STOP_FOR_SUBOPTIMUM_DOMAINS

PEANOHILBERT
WALLCLOCK
AUTO_SWAP_ENDIAN_READIC        # Enables automatic ENDIAN swapping for reading ICs

# WENDLAND_C6_KERNEL        # Switch to Wendland C6 kernel
# WC6_BIAS_CORRECTION       # Apply the bias correction for C6

MOREPARAMS
NOTYPEPREFIX_FFTW
ALLOWEXTRAPARAMS
HAVE_HDF5
# https://wwwmpa.mpa-garching.mpg.de/gadget/gadget-list/0432.html
H5_USE_16_API

# READ_HSML                     # reads hsml from IC file

DEBUG                     # enables core-dumps and FPU exceptions
# JD_VTURB                 # Compute vturb,vrms,vblk

# MAGNETIC
# TRACEDIVB
# MAGNETIC_SIGNALVEL
# MAGFORCE

# DIVBFORCE3=1.1
# AB_DIVBFORCE=0.25

# ARTIFICIAL_CONDUCTIVITY
# TIME_DEP_ART_COND=1.0
# AB_COND_GRAVITY=5.0

WAKEUP=3.0

# AB_SHEAR

# MAGNETIC_DISSIPATION
# TIME_DEP_MAGN_DISP
# AB_ART_DISP=1.0
# AB_SHOCK
# JD_SHOCK_RECONSTRUCT
