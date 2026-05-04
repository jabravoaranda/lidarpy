# -*- coding: utf-8 -*-
"""
Copyright 2016...2021 Volker Freudenthaler

Licensed under the EUPL, Version 1.1 only (the "Licence").

You may not use this work except in compliance with the Licence.
A copy of the licence is distributed with the code. Alternatively, you may obtain
a copy of the Licence at:

https://joinup.ec.europa.eu/community/eupl/og_page/eupl

Unless required by applicable law or agreed to in writing, software distributed
under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS
OF ANY KIND, either express or implied. See the Licence for the specific language governing
permissions and limitations under the Licence.

Equation reference: http://www.atmos-meas-tech-discuss.net/amt-2015-338/amt-2015-338.pdf
With equations code from Appendix C
Python 3.7, seaborn 0.9.0

Code description:

From measured lidar signals we cannot directly determine the desired backscatter coefficient (F11) and the linear depolarization ratio (LDR)
because of the cross talk between the channles and systematic errors of a lidar system.
http://www.atmos-meas-tech-discuss.net/amt-2015-338/amt-2015-338.pdf provides an analytical model for the description of these errors,
with which the measured signals can be corrected.
This code simulates the lidar measurements with "assumed true" model parameters from an input file, and calculates the correction parameters (G,H, and K).
The "assumed true" system parameters are the ones we think are the right ones, but in reality these parameters probably deviate from the assumed truth due to
uncertainties. The uncertainties of the "assumed true" parameters can be described in the input file. Then this code calculates the lidar signals and the
gain ratio eta* with all possible combinations of "errors", which represents the distribution of "possibly real" signals, and "corrects" them with the "assumed true"
GHK parameters (GT0, GR0, HT0, HR0, and K0) to derive finally the distributions of "possibly real" linear depolarization ratios (LDRCorr),
which are plotted for five different input linear depolarization ratios (LDRtrue). The red bars in the plots represent the input values of LDRtrue.
A complication arises from the fact that the correction parameter K = eta*/eta (Eq. 83) can depend on the LDR during the calibration measurement, i.e. LDRcal or aCal
in the code (see e.g. Eqs. (103), (115), and (141); mind the mistake in Eq. (116)). Therefor values of K for LDRcal = 0.004, 0.2, and 0.45 are calculated for
"assumed true" system parameters and printed in the output file behind the GH parameters. The full impact of the LDRcal dependent K can be considered in the error
calculation by specifying a range of possible LDRcal values in the input file. For the real calibration measurements a calibration range with low or no aerosol
content should be chosen, and the default in the input file is a range of LDRcal between 0.004 and 0.014 (i.e. 0.009 +-0.005).

Tip: In case you run the code with Spyder, all output text and plots can be displayed together in an IPython console, which can be saved as an html file.

Ver. 0.9.7:  includes the random error (signal noise) of the calibration and standard measurements
Changes:
    Line 1687   Eta = (TaR * TiR) / (TaT * TiT)
    Line 1691   K = Etax / Eta  # K of the real system; but correction in Line 1721 with K0 / Etax
    should work with nTCalT = nTCalR = 0
Ver. 0.9.7b:
    ToDo: include error due to TCalT und TCalR => determination of NCalT and NCalR etc. in error calculation line 1741ff
    combined error loops iNI and INCal for signals
Ver. 0.9.7c: individual error loops for each of the six signals
Ver. 0.9.7c2: different calculation of the signal noise errors
Ver. 0.9.7c3: n.a.different calculation of the signal noise errors
Ver. 0.9.7c4: test to speed up the loops for error calculation by moving them just before the actual calculation: still some code errors
Ver. 0.9.8:
    - correct calculation of Eta for cleaned anaylsers considering the combined transmission Eta = (TaT* TiT)(1 + cos2RotaT * DaT * DiT) and (TaR * TiR)(1 + cos2RotaR * DaR * DiR) according to the papers supplement Eqs. (S.10.10.1) ff
    - calculation of the PLDR from LDR and BSR, BSR, and LDRm
    - ND-filters can be added for the calibration measurements in the transmitted (TCalT) and the reflected path (TCalR) in order to include their uncertainties in the error calculation.
        The ND-filters effect is not included in the K-parameter. To do that, you have to multiply the eta, which is determined from the "calibration measurements with ND-filter", with the attenuation of the ND-filter "by hand", in order to get the calibration factor for the "standard measurements without ND-filter". Furthermore, the additional error of the calibration factor eta due to the uncertainty of the attenuation of the ND-filter has to be considered "by hand".
Ver. 0.9.8b:  change from  "TTa = TiT * TaT"  to  "TTa = TiT * TaT * ATPT" etc. (compare ver 0.9.8 with 0.9.8b) removes
	- the strong Tp dependence of the errors
	- the factor 2 in the GH parameters
    - see c:\technik\Optik\Polarizers\DepCal\ApplOpt\GH-parameters-190114.odt
Ver. 0.9.8c:  includes error of Etax
Ver. 0.9.8d:  Eta0, K0 etc in error loop replaced by Eta0y, K0y etc. Changes in signal noise calculations
Ver. 0.9.8e:  ambiguous laser spec. DOLP (no discrimination between left and right circular polarisation) replaced by Stokes parameters Qin, Uin
Ver. 0.9.8e2:  Added plot of LDRsim, Etax, Etapx, Etamx;  LDRCorr and aLDRcorr consistently named
Ver. 0.9.8e3:  Change of OutputFile name; Change of Ir and It noise if (CalcFrom0deg) = False;  (Different calculation of error contributions tested but not implemented)
Ver. 0.9.8e4:  text changed for y=+-1 (see line 274 ff and line 1044 ff
Ver. 0.9.8e5:  changed: LDRunCorr = LDRsim / Etax
Ver. 0.9.8e6:  K(0.05) instead K(0.02)
Ver. 0.9.8f:  Ioannis Binietoglou => for LINUX compatibility: Using os.path.join should work in all operating systems.
                # After line 1007:
                output_path = os.path.join('output_files', OutputFile)
                with open(output_path, 'w') as f:
                # Line 1147
                file = open(output_path, 'r')
Ver. 0.9.8g:  The LDRCal in the optic_input file is taken as first value for K(LDRCal)
Ver. 0.9.8h:  1. K(LDRCal) second order polynomial fit to values calculated for the listed LDRCal = 0.004 to 0.45 (without error calculation).
              2. Second order polynomial fits to upper and lower systematic error limits of the retrieved LDR depending on the true LDR.

 ========================================================
simulation: LDRsim = Ir / It with variable parameters (possible truths)
    G,H,Eta,Etax,K
    It = TaT * TiT * ATP1 * TiO * TiE * (GT + atrue * HT)
    LDRsim = Ir / It
consistency test: is forward simulation and correction consistent?
    LDRCorr = (LDRsim / Eta * (GT + HT) - (GR + HR)) / ((GR - HR) - LDRsim / Eta * (GT - HT)) => atrue?
assumed true: G0,H0,Eta0,Etax0,K0 => actual retrievals of LDRCorr
    => correct possible truths with assumed true G0,H0,Eta0
    measure: It, Ir, EtaX
    LDRunCorr = LDRsim / Etax
    correct it with G0,H0,K0:
    LDRCorr = (LDRsim / (Etax / K0) * (GT0 + HT0) - (GR0 + HR0)) / ((GR0 - HR0) - LDRsim0 / (Etax / K0) * (GT0 - HT0))
"""
# Comment:  The code might works with Python 2.7  with the help of following line, which enables Python2 to correctly interpret the Python 3 print statements.
from __future__ import print_function

# !/usr/bin/env python3

import os
from pathlib import Path
import sys

from scipy.stats import kurtosis
from scipy.stats import skew

# use: kurtosis(data, fisher=True,bias=False) => 0; skew(data,bias=False) => 0
# Comment: the seaborn library makes nicer plots, but the code works also without it.
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns_loaded = True
except ImportError:
    sns_loaded = False

# from time import clock # python 2
from timeit import default_timer as clock

# from matplotlib.backends.backend_pdf import PdfPages
# pdffile = '{}.pdf'.format('path')
# pp = PdfPages(pdffile)
## pp.savefig can be called multiple times to save to multiple pages
# pp.savefig()
# pp.close()

from contextlib import contextmanager

InputPath = Path(sys.argv[1])
OutputDir = Path(sys.argv[2])


@contextmanager
def redirect_stdout(new_target):
    old_target, sys.stdout = sys.stdout, new_target  # replace sys.stdout
    try:
        yield new_target  # run some code with the replaced stdout
    finally:
        sys.stdout.flush()
        sys.stdout = old_target  # restore to the previous value


"""
real_raw_input = vars(__builtins__).get('raw_input',input)
"""
try:
    import __builtin__

    input = getattr(__builtin__, "raw_input")
except (ImportError, AttributeError):
    pass

from distutils.util import strtobool


def user_yes_no_query(question):
    sys.stdout.write("%s [y/n]\n" % question)
    while True:
        try:
            return strtobool(input().lower())
        except ValueError:
            sys.stdout.write("Please respond with 'y' or 'n'.\n")


# if user_yes_no_query('want to exit?') == 1: sys.exit()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
fname = os.path.basename(abspath)
# os.chdir(dname)

# PrintToOutputFile = True

sqr05 = 0.5**0.5

# ---- Initial definition of variables; the actual values will be read in with exec(open('./optic_input.py').read()) below
# Do you want to calculate the errors? If not, just the GHK-parameters are determined.
Error_Calc = True
LID = "internal"
EID = "internal"
# --- IL Laser IL and +-Uncertainty
Qin, dQin, nQin = (
    1.0,
    0.0,
    0,
)  # second Stokes vector parameter; default 1 => linear polarization
Vin, dVin, nVin = 0.0, 0.0, 0  # fourth Stokes vector parameter
RotL, dRotL, nRotL = (
    0.0,
    0.0,
    1,
)  # alpha; rotation of laser polarization in degrees; default 0
# IL = 1e5      #photons in the laser beam, including detection efficiency of the telescope, atmodspheric and r^2 attenuation
# --- ME Emitter and +-Uncertainty
DiE, dDiE, nDiE = 0.0, 0.00, 1  # Diattenuation
TiE = 1.0  # Unpolarized transmittance
RetE, dRetE, nRetE = 0.0, 180.0, 0  # Retardance in degrees
RotE, dRotE, nRotE = 0.0, 0.0, 0  # beta: Rotation of optical element in degrees
# --- MO Receiver Optics including telescope
DiO, dDiO, nDiO = -0.055, 0.003, 1
TiO = 0.9
RetO, dRetO, nRetO = 0.0, 180.0, 2
RotO, dRotO, nRotO = 0.0, 0.1, 1  # gamma
# --- PBS MT transmitting path defined with (TS,TP);  and +-Uncertainty
TP, dTP, nTP = 0.98, 0.02, 1
TS, dTS, nTS = 0.001, 0.001, 1
TiT = 0.5 * (TP + TS)
DiT = (TP - TS) / (TP + TS)
# PolFilter
RetT, dRetT, nRetT = 0.0, 180.0, 0
ERaT, dERaT, nERaT = 0.001, 0.001, 1
RotaT, dRotaT, nRotaT = 0.0, 3.0, 1
DaT = (1 - ERaT) / (1 + ERaT)
TaT = 0.5 * (1 + ERaT)
# --- PBS MR reflecting path defined with (RS,RP);  and +-Uncertainty
RS_RP_depend_on_TS_TP = False
if RS_RP_depend_on_TS_TP:
    RP, dRP, nRP = 1 - TP, 0.0, 0
    RS, dRS, nRS = 1 - TS, 0.0, 0
else:
    RP, dRP, nRP = 0.05, 0.01, 1
    RS, dRS, nRS = 0.98, 0.01, 1
TiR = 0.5 * (RP + RS)
DiR = (RP - RS) / (RP + RS)
# PolFilter
RetR, dRetR, nRetR = 0.0, 180.0, 0
ERaR, dERaR, nERaR = 0.001, 0.001, 1
RotaR, dRotaR, nRotaR = 90.0, 3.0, 1
DaR = (1 - ERaR) / (1 + ERaR)
TaR = 0.5 * (1 + ERaR)

# +++ Orientation of the PBS with respect to the reference plane (see Polarisation-orientation.png and Polarisation-orientation-2.png in /system_settings)
#    Y = +1: PBS incidence plane is parallel to the reference plane and polarisation in reference plane is finally transmitted.
#    Y = -1: PBS incidence plane is perpendicular to the reference plane and polarisation in reference plane is finally reflected.
Y = 1.0

# Calibrator =  type defined by matrix values
LocC = 4  # location of calibrator: behind laser = 1; behind emitter = 2; before receiver = 3; before PBS = 4

# --- Additional attenuation (transmission of the ND-filter) during the calibration
TCalT, dTCalT, nTCalT = 1, 0.0, 0  # transmitting path; error calc not working yet
TCalR, dTCalR, nTCalR = 1, 0.0, 0  # reflecting path; error calc not working yet

# *** signal noise error calculation
#   --- number of photon counts in the signal summed up in the calibration range during the calibration measurements
NCalT = 1e6  # default 1e6, assumed the same in +45° and -45° signals
NCalR = 1e6  # default 1e6, assumed the same in +45° and -45° signals
NILfac = (
    1.0  # duration of standard (0°) measurement relative to calibration measurements
)
nNCal = 0  # error nNCal: one-sigma in steps to left and right for calibration signals
nNI = 0  # error nNI: one-sigma in steps to left and right for 0° signals
NI = 50000  # number of photon counts in the parallel 0°-signal
eFacT = 1.0  # rel. amplification of transmitted channel, approximate values are sufficient; def. = 1
eFacR = 10.0
IoutTp0, IoutTp, dIoutTp0 = 0.5, 0.5, 0.0
IoutTm0, IoutTm, dIoutTm0 = 0.5, 0.5, 0.0
IoutRp0, IoutRp, dIoutRp0 = 0.5, 0.5, 0.0
IoutRm0, IoutRm, dIoutRm0 = 0.5, 0.5, 0.0
It0, It, dIt0 = 1, 1, 0
Ir0, Ir, dTr0 = 1, 1, 0
CalcFrom0deg = True

TypeC = 3  # linear polarizer calibrator
# example with extinction ratio 0.001
DiC, dDiC, nDiC = 1.0, 0.0, 0  # ideal 1.0
TiC = 0.5  # ideal 0.5
RetC, dRetC, nRetC = 0.0, 0.0, 0
RotC, dRotC, nRotC = 0.0, 0.1, 0  # constant calibrator offset epsilon
RotationErrorEpsilonForNormalMeasurements = (
    False  # is in general False for TypeC == 3 calibrator
)

# Rotation error without calibrator: if False, then epsilon = 0 for normal measurements
RotationErrorEpsilonForNormalMeasurements = True
# BSR backscatter ratio
# BSR, dBSR, nBSR = 10, 0.05, 1
BSR = np.zeros(5)
BSR = [1.1, 2, 5, 10.0, 50.0]
# theoretical molecular LDR  LDRm
LDRm, dLDRm, nLDRm = 0.004, 0.001, 1
# LDRCal assumed atmospheric linear depolarization ratio during the calibration measurements (first guess)
LDRCal0, dLDRCal, nLDRCal = 0.25, 0.04, 1
LDRCal = LDRCal0
# measured LDRm will be corrected with calculated parameters
LDRmeas = 0.015
# LDRtrue for simulation of measurement => LDRsim
LDRtrue = 0.004
LDRtrue2 = 0.004
LDRunCorr = 1.0
# Initialize other values to 0
ER, nER, dER = 0.001, 0, 0.001
K = 0.0
Km = 0.0
Kp = 0.0
LDRCorr = 0.0
Eta = 0.0
Ir = 0.0
It = 0.0
h = 1.0

Loc = [
    "",
    "behind the laser",
    "behind the emitter",
    "before the receiver",
    "before the PBS",
]
Type = [
    "",
    "Mechanical rotator",
    "HWP rotator",
    "Linear polarizer",
    "QWP rotator",
    "Circular polarizer",
    "real HWP +-22.5°",
]

bPlotEtax = False

#  end of initial definition of variables
# *******************************************************************************************************************************
# ---- Read actual lidar system parameters from optic_input.py  (must be in the programs sub-directory 'system_settings')
# *******************************************************************************************************************************
# # InputFile = 'optic_input_0.9.8e4-PollyXT_Lacros.py'
# InputFile = 'optic_input_example_lidar_ver0.9.8e.py'
# InputFile = 'calibrator-test-1-ver0.9.8e.py'
# InputFile = 'optic_input_0.9.8e4-BRC-532.py'
# InputFile = ''
# InputFile = ''
# InputFile = ''
# InputFile = ''
# InputFile = ''
# InputFile = 'CIEMAT-200929-test-ver0.9.8f.py'
# InputFile = 'calibrator-HWP-retard-error-ver0.9.8e.py'
# InputFile = 'MUSA C1A-ver0.98e.py'
# InputFile = 'EVE-210411-ver0.9.8f.py'
# InputFile = 'Cilbolton-210609-ver0.9.8f.py'
# InputFile = 'optic_input_0.9.8e4-PollyXT_Cyprus_355_210115_vf_4.py'
# InputFile = 'MUSA C1B-ver0.98e.py'
# InputFile = 'RALI-may2020-X.py'
# InputFile = 'optic_input_0.9.8h-PollyXT_Cyprus_532_210429_vf_4.py'
# print(f"InputFile is: {InputFile}")
# InputFile = 'optic_input_mhc_rot_20220614.py' #FIXME: change this for a environment variable from run_GHK_simulator.py
# InputFile = ''

# *******************************************************************************************************************************

"""
print("From ", dname)
print("Running ", fname)
print("Reading input file ", InputFile, " for")
"""
# input_path = os.path.join(".", "system_settings", InputFile)
# this works with Python 2 and 3!
exec(open(InputPath).read(), globals())
#  end of read actual system parameters


# --- Manual Parameter Change ---
#  (use for quick parameter changes without changing the input file )
# DiO = 0.
# LDRtrue = 0.45
# LDRtrue2 = 0.004
# Y = -1
# LocC = 4 #location of calibrator: 1 = behind laser; 2 = behind emitter; 3 = before receiver; 4 = before PBS
# #TypeC = 6  Don't change the TypeC here
# RotationErrorEpsilonForNormalMeasurements = True
# LDRCal = 0.25
# # --- Errors
Qin0, dQin, nQin = Qin, dQin, nQin
Vin0, dVin, nVin = Vin, dVin, nVin
RotL0, dRotL, nRotL = RotL, dRotL, nRotL

DiE0, dDiE, nDiE = DiE, dDiE, nDiE
RetE0, dRetE, nRetE = RetE, dRetE, nRetE
RotE0, dRotE, nRotE = RotE, dRotE, nRotE

DiO0, dDiO, nDiO = DiO, dDiO, nDiO
RetO0, dRetO, nRetO = RetO, dRetO, nRetO
RotO0, dRotO, nRotO = RotO, dRotO, nRotO

DiC0, dDiC, nDiC = DiC, dDiC, nDiC
RetC0, dRetC, nRetC = RetC, dRetC, nRetC
RotC0, dRotC, nRotC = RotC, dRotC, nRotC

TP0, dTP, nTP = TP, dTP, nTP
TS0, dTS, nTS = TS, dTS, nTS
RetT0, dRetT, nRetT = RetT, dRetT, nRetT

ERaT0, dERaT, nERaT = ERaT, dERaT, nERaT
RotaT0, dRotaT, nRotaT = RotaT, dRotaT, nRotaT

RP0, dRP, nRP = RP, dRP, nRP
RS0, dRS, nRS = RS, dRS, nRS
RetR0, dRetR, nRetR = RetR, dRetR, nRetR

ERaR0, dERaR, nERaR = ERaR, dERaR, nERaR
RotaR0, dRotaR, nRotaR = RotaR, dRotaR, nRotaR

LDRCal0, dLDRCal, nLDRCal = LDRCal, dLDRCal, nLDRCal

# BSR0, dBSR, nBSR = BSR, dBSR, nBSR
LDRm0, dLDRm, nLDRm = LDRm, dLDRm, nLDRm
# ---------- End of manual parameter change

RotL, RotE, RetE, DiE, RotO, RetO, DiO, RotC, RetC, DiC = (
    RotL0,
    RotE0,
    RetE0,
    DiE0,
    RotO0,
    RetO0,
    DiO0,
    RotC0,
    RetC0,
    DiC0,
)
TP, TS, RP, RS, ERaT, RotaT, RetT, ERaR, RotaR, RetR = (
    TP0,
    TS0,
    RP0,
    RS0,
    ERaT0,
    RotaT0,
    RetT0,
    ERaR0,
    RotaR0,
    RetR0,
)
LDRCal = LDRCal0
DTa0, TTa0, DRa0, TRa0, LDRsimx, LDRCorr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
TCalT0, TCalR0 = TCalT, TCalR

TiT = 0.5 * (TP + TS)
DiT = (TP - TS) / (TP + TS)
ZiT = (1.0 - DiT**2) ** 0.5
TiR = 0.5 * (RP + RS)
DiR = (RP - RS) / (RP + RS)
ZiR = (1.0 - DiR**2) ** 0.5

C2aT = np.cos(np.deg2rad(2.0 * RotaT))
C2aR = np.cos(np.deg2rad(2.0 * RotaR))
ATPT = float(1.0 + C2aT * DaT * DiT)
ARPT = float(1.0 + C2aR * DaR * DiR)
TTa = TiT * TaT * ATPT  # unpolarized transmission
TRa = TiR * TaR * ARPT  # unpolarized transmission
Eta0 = TRa / TTa

# --- alternative texts for output
dY = ["perpendicular", "", "parallel"]
dY2 = ["reflected", "", "transmitted"]
if (abs(RotL) < 45 and Y == 1) or (abs(RotL) >= 45 and Y == -1):
    dY3 = "the parallel laser polarisation is detected in the transmitted channel."
else:
    dY3 = "the parallel laser polarisation is detected in the reflected channel."

# --- check input errors
if ((Qin**2 + Vin**2) ** 0.5) > 1:
    print("Error: degree of polarisation of laser > 1. Check Qin and Vin! ")
    sys.exit()

# --- this subroutine is for the calculation of the PLDR from LDR, BSR, and LDRm -------------------
def CalcPLDR(LDR, BSR, LDRm):
    PLDR = (BSR * (1.0 + LDRm) * LDR - LDRm * (1.0 + LDR)) / (
        BSR * (1.0 + LDRm) - (1.0 + LDR)
    )
    return PLDR


# --- this subroutine is for the calculation with certain fixed parameters ------------------------
def Calc(
    TCalT,
    TCalR,
    NCalT,
    NCalR,
    Qin,
    Vin,
    RotL,
    RotE,
    RetE,
    DiE,
    RotO,
    RetO,
    DiO,
    RotC,
    RetC,
    DiC,
    TP,
    TS,
    RP,
    RS,
    ERaT,
    RotaT,
    RetT,
    ERaR,
    RotaR,
    RetR,
    LDRCal,
):
    # ---- Do the calculations of bra-ket vectors
    h = -1.0 if TypeC == 2 else 1
    # from input file:  assumed LDRCal for calibration measurements
    aCal = (1.0 - LDRCal) / (1.0 + LDRCal)
    atrue = (1.0 - LDRtrue) / (1.0 + LDRtrue)

    # angles of emitter and laser and calibrator and receiver optics
    # RotL = alpha, RotE = beta, RotO = gamma, RotC = epsilon
    S2a = np.sin(2 * np.deg2rad(RotL))
    C2a = np.cos(2 * np.deg2rad(RotL))
    S2b = np.sin(2 * np.deg2rad(RotE))
    C2b = np.cos(2 * np.deg2rad(RotE))
    S2ab = np.sin(np.deg2rad(2 * RotL - 2 * RotE))
    C2ab = np.cos(np.deg2rad(2 * RotL - 2 * RotE))
    S2g = np.sin(np.deg2rad(2 * RotO))
    C2g = np.cos(np.deg2rad(2 * RotO))

    # Laser with Degree of linear polarization DOLP
    IinL = 1.0
    QinL = Qin
    UinL = 0.0
    VinL = Vin
    # VinL = (1. - DOLP ** 2) ** 0.5

    # Stokes Input Vector rotation Eq. E.4
    A = C2a * QinL - S2a * UinL
    B = S2a * QinL + C2a * UinL
    # Stokes Input Vector rotation Eq. E.9
    C = C2ab * QinL - S2ab * UinL
    D = S2ab * QinL + C2ab * UinL

    # emitter optics
    CosE = np.cos(np.deg2rad(RetE))
    SinE = np.sin(np.deg2rad(RetE))
    ZiE = (1.0 - DiE**2) ** 0.5
    WiE = 1.0 - ZiE * CosE

    # Stokes Input Vector after emitter optics equivalent to Eq. E.9 with already rotated input vector from Eq. E.4
    # b = beta
    IinE = IinL + DiE * C
    QinE = C2b * DiE * IinL + A + S2b * (WiE * D - ZiE * SinE * VinL)
    UinE = S2b * DiE * IinL + B - C2b * (WiE * D - ZiE * SinE * VinL)
    VinE = -ZiE * SinE * D + ZiE * CosE * VinL

    # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
    IinF = IinE
    QinF = aCal * QinE
    UinF = -aCal * UinE
    VinF = (1.0 - 2.0 * aCal) * VinE

    # receiver optics
    CosO = np.cos(np.deg2rad(RetO))
    SinO = np.sin(np.deg2rad(RetO))
    ZiO = (1.0 - DiO**2) ** 0.5
    WiO = 1.0 - ZiO * CosO

    # calibrator
    CosC = np.cos(np.deg2rad(RetC))
    SinC = np.sin(np.deg2rad(RetC))
    ZiC = (1.0 - DiC**2) ** 0.5
    WiC = 1.0 - ZiC * CosC

    # Stokes Input Vector before the polarising beam splitter Eq. E.31
    A = C2g * QinE - S2g * UinE
    B = S2g * QinE + C2g * UinE

    IinP = IinE + DiO * aCal * A
    QinP = (
        C2g * DiO * IinE
        + aCal * QinE
        - S2g * (WiO * aCal * B + ZiO * SinO * (1.0 - 2.0 * aCal) * VinE)
    )
    UinP = (
        S2g * DiO * IinE
        - aCal * UinE
        + C2g * (WiO * aCal * B + ZiO * SinO * (1.0 - 2.0 * aCal) * VinE)
    )
    VinP = ZiO * SinO * aCal * B + ZiO * CosO * (1.0 - 2.0 * aCal) * VinE

    # -------------------------
    # F11 assuemd to be = 1  => measured: F11m = IinP / IinE with atrue
    # F11sim = TiO*(IinE + DiO*atrue*A)/IinE
    # -------------------------

    # analyser
    if RS_RP_depend_on_TS_TP:
        RS = 1.0 - TS
        RP = 1.0 - TP

    TiT = 0.5 * (TP + TS)
    DiT = (TP - TS) / (TP + TS)
    ZiT = (1.0 - DiT**2) ** 0.5
    TiR = 0.5 * (RP + RS)
    DiR = (RP - RS) / (RP + RS)
    ZiR = (1.0 - DiR**2) ** 0.5
    CosT = np.cos(np.deg2rad(RetT))
    SinT = np.sin(np.deg2rad(RetT))
    CosR = np.cos(np.deg2rad(RetR))
    SinR = np.sin(np.deg2rad(RetR))

    DaT = (1.0 - ERaT) / (1.0 + ERaT)
    DaR = (1.0 - ERaR) / (1.0 + ERaR)
    TaT = 0.5 * (1.0 + ERaT)
    TaR = 0.5 * (1.0 + ERaR)

    S2aT = np.sin(np.deg2rad(h * 2 * RotaT))
    C2aT = np.cos(np.deg2rad(2 * RotaT))
    S2aR = np.sin(np.deg2rad(h * 2 * RotaR))
    C2aR = np.cos(np.deg2rad(2 * RotaR))

    # Analyzer As before the PBS Eq. D.5; combined PBS and cleaning pol-filter
    ATPT = 1.0 + C2aT * DaT * DiT  # unpolarized transmission correction
    TTa = TiT * TaT * ATPT  # unpolarized transmission
    ATP1 = 1.0
    ATP2 = Y * (DiT + C2aT * DaT) / ATPT
    ATP3 = Y * S2aT * DaT * ZiT * CosT / ATPT
    ATP4 = S2aT * DaT * ZiT * SinT / ATPT
    ATP = np.array([ATP1, ATP2, ATP3, ATP4])
    DTa = ATP2 * Y

    ARPT = 1 + C2aR * DaR * DiR  # unpolarized transmission correction
    TRa = TiR * TaR * ARPT  # unpolarized transmission
    ARP1 = 1
    ARP2 = Y * (DiR + C2aR * DaR) / ARPT
    ARP3 = Y * S2aR * DaR * ZiR * CosR / ARPT
    ARP4 = S2aR * DaR * ZiR * SinR / ARPT
    ARP = np.array([ARP1, ARP2, ARP3, ARP4])
    DRa = ARP2 * Y

    # ---- Calculate signals and correction parameters for diffeent locations and calibrators
    if LocC == 4:  # Calibrator before the PBS
        # print("Calibrator location not implemented yet")

        # S2ge = np.sin(np.deg2rad(2*RotO + h*2*RotC))
        # C2ge = np.cos(np.deg2rad(2*RotO + h*2*RotC))
        S2e = np.sin(np.deg2rad(h * 2 * RotC))
        C2e = np.cos(np.deg2rad(2 * RotC))
        # rotated AinP by epsilon Eq. C.3
        ATP2e = C2e * ATP2 + S2e * ATP3
        ATP3e = C2e * ATP3 - S2e * ATP2
        ARP2e = C2e * ARP2 + S2e * ARP3
        ARP3e = C2e * ARP3 - S2e * ARP2
        ATPe = np.array([ATP1, ATP2e, ATP3e, ATP4])
        ARPe = np.array([ARP1, ARP2e, ARP3e, ARP4])
        # Stokes Input Vector before the polarising beam splitter Eq. E.31
        A = C2g * QinE - S2g * UinE
        B = S2g * QinE + C2g * UinE
        # C = (WiO*aCal*B + ZiO*SinO*(1-2*aCal)*VinE)
        Co = ZiO * SinO * VinE
        Ca = WiO * B - 2 * ZiO * SinO * VinE
        # C = Co + aCal*Ca
        # IinP = (IinE + DiO*aCal*A)
        # QinP = (C2g*DiO*IinE + aCal*QinE - S2g*C)
        # UinP = (S2g*DiO*IinE - aCal*UinE + C2g*C)
        # VinP = (ZiO*SinO*aCal*B + ZiO*CosO*(1-2*aCal)*VinE)
        IinPo = IinE
        QinPo = C2g * DiO * IinE - S2g * Co
        UinPo = S2g * DiO * IinE + C2g * Co
        VinPo = ZiO * CosO * VinE

        IinPa = DiO * A
        QinPa = QinE - S2g * Ca
        UinPa = -UinE + C2g * Ca
        VinPa = ZiO * (SinO * B - 2 * CosO * VinE)

        IinP = IinPo + aCal * IinPa
        QinP = QinPo + aCal * QinPa
        UinP = UinPo + aCal * UinPa
        VinP = VinPo + aCal * VinPa
        # Stokes Input Vector before the polarising beam splitter rotated by epsilon Eq. C.3
        # QinPe = C2e*QinP + S2e*UinP
        # UinPe = C2e*UinP - S2e*QinP
        QinPoe = C2e * QinPo + S2e * UinPo
        UinPoe = C2e * UinPo - S2e * QinPo
        QinPae = C2e * QinPa + S2e * UinPa
        UinPae = C2e * UinPa - S2e * QinPa
        QinPe = C2e * QinP + S2e * UinP
        UinPe = C2e * UinP - S2e * QinP

        # Calibration signals and Calibration correction K from measurements with LDRCal / aCal
        if (TypeC == 2) or (TypeC == 1):  # rotator calibration Eq. C.4
            # parameters for calibration with aCal
            AT = ATP1 * IinP + h * ATP4 * VinP
            BT = ATP3e * QinP - h * ATP2e * UinP
            AR = ARP1 * IinP + h * ARP4 * VinP
            BR = ARP3e * QinP - h * ARP2e * UinP
            # Correction parameters for normal measurements; they are independent of LDR
            if not RotationErrorEpsilonForNormalMeasurements:  # calibrator taken out
                IS1 = np.array([IinPo, QinPo, UinPo, VinPo])
                IS2 = np.array([IinPa, QinPa, UinPa, VinPa])
                GT = np.dot(ATP, IS1)
                GR = np.dot(ARP, IS1)
                HT = np.dot(ATP, IS2)
                HR = np.dot(ARP, IS2)
            else:
                IS1 = np.array([IinPo, QinPo, UinPo, VinPo])
                IS2 = np.array([IinPa, QinPa, UinPa, VinPa])
                GT = np.dot(ATPe, IS1)
                GR = np.dot(ARPe, IS1)
                HT = np.dot(ATPe, IS2)
                HR = np.dot(ARPe, IS2)
        elif (TypeC == 3) or (TypeC == 4):  # linear polariser calibration Eq. C.5
            # parameters for calibration with aCal
            AT = (
                ATP1 * IinP + ATP3e * UinPe + ZiC * CosC * (ATP2e * QinPe + ATP4 * VinP)
            )
            BT = DiC * (ATP1 * UinPe + ATP3e * IinP) - ZiC * SinC * (
                ATP2e * VinP - ATP4 * QinPe
            )
            AR = (
                ARP1 * IinP + ARP3e * UinPe + ZiC * CosC * (ARP2e * QinPe + ARP4 * VinP)
            )
            BR = DiC * (ARP1 * UinPe + ARP3e * IinP) - ZiC * SinC * (
                ARP2e * VinP - ARP4 * QinPe
            )
            # Correction parameters for normal measurements; they are independent of LDR
            if not RotationErrorEpsilonForNormalMeasurements:  # calibrator taken out
                IS1 = np.array([IinPo, QinPo, UinPo, VinPo])
                IS2 = np.array([IinPa, QinPa, UinPa, VinPa])
                GT = np.dot(ATP, IS1)
                GR = np.dot(ARP, IS1)
                HT = np.dot(ATP, IS2)
                HR = np.dot(ARP, IS2)
            else:
                IS1e = np.array(
                    [
                        IinPo + DiC * QinPoe,
                        DiC * IinPo + QinPoe,
                        ZiC * (CosC * UinPoe + SinC * VinPo),
                        -ZiC * (SinC * UinPoe - CosC * VinPo),
                    ]
                )
                IS2e = np.array(
                    [
                        IinPa + DiC * QinPae,
                        DiC * IinPa + QinPae,
                        ZiC * (CosC * UinPae + SinC * VinPa),
                        -ZiC * (SinC * UinPae - CosC * VinPa),
                    ]
                )
                GT = np.dot(ATPe, IS1e)
                GR = np.dot(ARPe, IS1e)
                HT = np.dot(ATPe, IS2e)
                HR = np.dot(ARPe, IS2e)
        elif (
            TypeC == 6
        ):  # diattenuator calibration +-22.5° rotated_diattenuator_X22x5deg.odt
            # parameters for calibration with aCal
            AT = (
                ATP1 * IinP
                + sqr05 * DiC * (ATP1 * QinPe + ATP2e * IinP)
                + (1.0 - 0.5 * WiC) * (ATP2e * QinPe + ATP3e * UinPe)
                + ZiC
                * (sqr05 * SinC * (ATP3e * VinP - ATP4 * UinPe) + ATP4 * CosC * VinP)
            )
            BT = (
                sqr05 * DiC * (ATP1 * UinPe + ATP3e * IinP)
                + 0.5 * WiC * (ATP2e * UinPe + ATP3e * QinPe)
                - sqr05 * ZiC * SinC * (ATP2e * VinP - ATP4 * QinPe)
            )
            AR = (
                ARP1 * IinP
                + sqr05 * DiC * (ARP1 * QinPe + ARP2e * IinP)
                + (1.0 - 0.5 * WiC) * (ARP2e * QinPe + ARP3e * UinPe)
                + ZiC
                * (sqr05 * SinC * (ARP3e * VinP - ARP4 * UinPe) + ARP4 * CosC * VinP)
            )
            BR = (
                sqr05 * DiC * (ARP1 * UinPe + ARP3e * IinP)
                + 0.5 * WiC * (ARP2e * UinPe + ARP3e * QinPe)
                - sqr05 * ZiC * SinC * (ARP2e * VinP - ARP4 * QinPe)
            )
            # Correction parameters for normal measurements; they are independent of LDR
            if not RotationErrorEpsilonForNormalMeasurements:  # calibrator taken out
                IS1 = np.array([IinPo, QinPo, UinPo, VinPo])
                IS2 = np.array([IinPa, QinPa, UinPa, VinPa])
                GT = np.dot(ATP, IS1)
                GR = np.dot(ARP, IS1)
                HT = np.dot(ATP, IS2)
                HR = np.dot(ARP, IS2)
            else:
                IS1e = np.array(
                    [
                        IinPo + DiC * QinPoe,
                        DiC * IinPo + QinPoe,
                        ZiC * (CosC * UinPoe + SinC * VinPo),
                        -ZiC * (SinC * UinPoe - CosC * VinPo),
                    ]
                )
                IS2e = np.array(
                    [
                        IinPa + DiC * QinPae,
                        DiC * IinPa + QinPae,
                        ZiC * (CosC * UinPae + SinC * VinPa),
                        -ZiC * (SinC * UinPae - CosC * VinPa),
                    ]
                )
                GT = np.dot(ATPe, IS1e)
                GR = np.dot(ARPe, IS1e)
                HT = np.dot(ATPe, IS2e)
                HR = np.dot(ARPe, IS2e)
        else:
            print("Calibrator not implemented yet")
            sys.exit()

    elif LocC == 3:  # C before receiver optics Eq.57

        # S2ge = np.sin(np.deg2rad(2*RotO - 2*RotC))
        # C2ge = np.cos(np.deg2rad(2*RotO - 2*RotC))
        S2e = np.sin(np.deg2rad(2.0 * RotC))
        C2e = np.cos(np.deg2rad(2.0 * RotC))

        # As with C before the receiver optics (rotated_diattenuator_X22x5deg.odt)
        AF1 = np.array([1.0, C2g * DiO, S2g * DiO, 0.0])
        AF2 = np.array(
            [C2g * DiO, 1.0 - S2g**2 * WiO, S2g * C2g * WiO, -S2g * ZiO * SinO]
        )
        AF3 = np.array(
            [S2g * DiO, S2g * C2g * WiO, 1.0 - C2g**2 * WiO, C2g * ZiO * SinO]
        )
        AF4 = np.array([0.0, S2g * SinO, -C2g * SinO, CosO])

        ATF = ATP1 * AF1 + ATP2 * AF2 + ATP3 * AF3 + ATP4 * AF4
        ARF = ARP1 * AF1 + ARP2 * AF2 + ARP3 * AF3 + ARP4 * AF4
        ATF2 = ATF[1]
        ATF3 = ATF[2]
        ARF2 = ARF[1]
        ARF3 = ARF[2]

        # rotated AinF by epsilon
        ATF1 = ATF[0]
        ATF4 = ATF[3]
        ATF2e = C2e * ATF[1] + S2e * ATF[2]
        ATF3e = C2e * ATF[2] - S2e * ATF[1]
        ARF1 = ARF[0]
        ARF4 = ARF[3]
        ARF2e = C2e * ARF[1] + S2e * ARF[2]
        ARF3e = C2e * ARF[2] - S2e * ARF[1]

        ATFe = np.array([ATF1, ATF2e, ATF3e, ATF4])
        ARFe = np.array([ARF1, ARF2e, ARF3e, ARF4])

        QinEe = C2e * QinE + S2e * UinE
        UinEe = C2e * UinE - S2e * QinE

        # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
        IinF = IinE
        QinF = aCal * QinE
        UinF = -aCal * UinE
        VinF = (1.0 - 2.0 * aCal) * VinE

        IinFo = IinE
        QinFo = 0.0
        UinFo = 0.0
        VinFo = VinE

        IinFa = 0.0
        QinFa = QinE
        UinFa = -UinE
        VinFa = -2.0 * VinE

        # Stokes Input Vector before receiver optics rotated by epsilon Eq. C.3
        QinFe = C2e * QinF + S2e * UinF
        UinFe = C2e * UinF - S2e * QinF
        QinFoe = C2e * QinFo + S2e * UinFo
        UinFoe = C2e * UinFo - S2e * QinFo
        QinFae = C2e * QinFa + S2e * UinFa
        UinFae = C2e * UinFa - S2e * QinFa

        # Calibration signals and Calibration correction K from measurements with LDRCal / aCal
        if (TypeC == 2) or (TypeC == 1):  # rotator calibration Eq. C.4
            # parameters for calibration with aCal
            AT = ATF1 * IinF + ATF4 * h * VinF
            BT = ATF3e * QinF - ATF2e * h * UinF
            AR = ARF1 * IinF + ARF4 * h * VinF
            BR = ARF3e * QinF - ARF2e * h * UinF
            # Correction parameters for normal measurements; they are independent of LDR
            if not RotationErrorEpsilonForNormalMeasurements:
                GT = ATF1 * IinE + ATF4 * VinE
                GR = ARF1 * IinE + ARF4 * VinE
                HT = ATF2 * QinE - ATF3 * UinE - ATF4 * 2 * VinE
                HR = ARF2 * QinE - ARF3 * UinE - ARF4 * 2 * VinE
            else:
                GT = ATF1 * IinE + ATF4 * h * VinE
                GR = ARF1 * IinE + ARF4 * h * VinE
                HT = ATF2e * QinE - ATF3e * h * UinE - ATF4 * h * 2 * VinE
                HR = ARF2e * QinE - ARF3e * h * UinE - ARF4 * h * 2 * VinE
        elif (TypeC == 3) or (TypeC == 4):  # linear polariser calibration Eq. C.5
            # p = +45°, m = -45°
            IF1e = np.array([IinF, ZiC * CosC * QinFe, UinFe, ZiC * CosC * VinF])
            IF2e = np.array(
                [DiC * UinFe, -ZiC * SinC * VinF, DiC * IinF, ZiC * SinC * QinFe]
            )
            AT = np.dot(ATFe, IF1e)
            AR = np.dot(ARFe, IF1e)
            BT = np.dot(ATFe, IF2e)
            BR = np.dot(ARFe, IF2e)

            # Correction parameters for normal measurements; they are independent of LDR  --- the same as for TypeC = 6
            if not RotationErrorEpsilonForNormalMeasurements:  # calibrator taken out
                IS1 = np.array([IinE, 0.0, 0.0, VinE])
                IS2 = np.array([0.0, QinE, -UinE, -2.0 * VinE])
                GT = np.dot(ATF, IS1)
                GR = np.dot(ARF, IS1)
                HT = np.dot(ATF, IS2)
                HR = np.dot(ARF, IS2)
            else:
                IS1e = np.array(
                    [
                        IinFo + DiC * QinFoe,
                        DiC * IinFo + QinFoe,
                        ZiC * (CosC * UinFoe + SinC * VinFo),
                        -ZiC * (SinC * UinFoe - CosC * VinFo),
                    ]
                )
                IS2e = np.array(
                    [
                        IinFa + DiC * QinFae,
                        DiC * IinFa + QinFae,
                        ZiC * (CosC * UinFae + SinC * VinFa),
                        -ZiC * (SinC * UinFae - CosC * VinFa),
                    ]
                )
                GT = np.dot(ATFe, IS1e)
                GR = np.dot(ARFe, IS1e)
                HT = np.dot(ATFe, IS2e)
                HR = np.dot(ARFe, IS2e)

        elif (
            TypeC == 6
        ):  # diattenuator calibration +-22.5° rotated_diattenuator_X22x5deg.odt
            # parameters for calibration with aCal
            IF1e = np.array(
                [
                    IinF + sqr05 * DiC * QinFe,
                    sqr05 * DiC * IinF + (1.0 - 0.5 * WiC) * QinFe,
                    (1.0 - 0.5 * WiC) * UinFe + sqr05 * ZiC * SinC * VinF,
                    -sqr05 * ZiC * SinC * UinFe + ZiC * CosC * VinF,
                ]
            )
            IF2e = np.array(
                [
                    sqr05 * DiC * UinFe,
                    0.5 * WiC * UinFe - sqr05 * ZiC * SinC * VinF,
                    sqr05 * DiC * IinF + 0.5 * WiC * QinFe,
                    sqr05 * ZiC * SinC * QinFe,
                ]
            )
            AT = np.dot(ATFe, IF1e)
            AR = np.dot(ARFe, IF1e)
            BT = np.dot(ATFe, IF2e)
            BR = np.dot(ARFe, IF2e)

            # Correction parameters for normal measurements; they are independent of LDR
            if not RotationErrorEpsilonForNormalMeasurements:  # calibrator taken out
                # IS1 = np.array([IinE,0,0,VinE])
                # IS2 = np.array([0,QinE,-UinE,-2*VinE])
                IS1 = np.array([IinFo, 0.0, 0.0, VinFo])
                IS2 = np.array([0.0, QinFa, UinFa, VinFa])
                GT = np.dot(ATF, IS1)
                GR = np.dot(ARF, IS1)
                HT = np.dot(ATF, IS2)
                HR = np.dot(ARF, IS2)
            else:
                IS1e = np.array(
                    [
                        IinFo + DiC * QinFoe,
                        DiC * IinFo + QinFoe,
                        ZiC * (CosC * UinFoe + SinC * VinFo),
                        -ZiC * (SinC * UinFoe - CosC * VinFo),
                    ]
                )
                IS2e = np.array(
                    [
                        IinFa + DiC * QinFae,
                        DiC * IinFa + QinFae,
                        ZiC * (CosC * UinFae + SinC * VinFa),
                        -ZiC * (SinC * UinFae - CosC * VinFa),
                    ]
                )
                # IS1e = np.array([IinFo,0,0,VinFo])
                # IS2e = np.array([0,QinFae,UinFae,VinFa])
                GT = np.dot(ATFe, IS1e)
                GR = np.dot(ARFe, IS1e)
                HT = np.dot(ATFe, IS2e)
                HR = np.dot(ARFe, IS2e)

        else:
            print("Calibrator not implemented yet")
            sys.exit()

    elif (
        LocC == 2
    ):  # C behind emitter optics Eq.57 -------------------------------------------------------
        # print("Calibrator location not implemented yet")
        S2e = np.sin(np.deg2rad(2.0 * RotC))
        C2e = np.cos(np.deg2rad(2.0 * RotC))

        # AS with C before the receiver optics (see document rotated_diattenuator_X22x5deg.odt)
        AF1 = np.array([1, C2g * DiO, S2g * DiO, 0.0])
        AF2 = np.array(
            [C2g * DiO, 1.0 - S2g**2 * WiO, S2g * C2g * WiO, -S2g * ZiO * SinO]
        )
        AF3 = np.array(
            [S2g * DiO, S2g * C2g * WiO, 1.0 - C2g**2 * WiO, C2g * ZiO * SinO]
        )
        AF4 = np.array([0.0, S2g * SinO, -C2g * SinO, CosO])

        ATF = ATP1 * AF1 + ATP2 * AF2 + ATP3 * AF3 + ATP4 * AF4
        ARF = ARP1 * AF1 + ARP2 * AF2 + ARP3 * AF3 + ARP4 * AF4
        ATF1 = ATF[0]
        ATF2 = ATF[1]
        ATF3 = ATF[2]
        ATF4 = ATF[3]
        ARF1 = ARF[0]
        ARF2 = ARF[1]
        ARF3 = ARF[2]
        ARF4 = ARF[3]

        # AS with C behind the emitter
        # terms without aCal
        ATE1o, ARE1o = ATF1, ARF1
        ATE2o, ARE2o = 0.0, 0.0
        ATE3o, ARE3o = 0.0, 0.0
        ATE4o, ARE4o = ATF4, ARF4
        # terms with aCal
        ATE1a, ARE1a = 0.0, 0.0
        ATE2a, ARE2a = ATF2, ARF2
        ATE3a, ARE3a = -ATF3, -ARF3
        ATE4a, ARE4a = -2.0 * ATF4, -2.0 * ARF4
        # rotated AinEa by epsilon
        ATE2ae = C2e * ATF2 + S2e * ATF3
        ATE3ae = -S2e * ATF2 - C2e * ATF3
        ARE2ae = C2e * ARF2 + S2e * ARF3
        ARE3ae = -S2e * ARF2 - C2e * ARF3

        ATE1 = ATE1o
        ATE2e = aCal * ATE2ae
        ATE3e = aCal * ATE3ae
        ATE4 = (1 - 2 * aCal) * ATF4
        ARE1 = ARE1o
        ARE2e = aCal * ARE2ae
        ARE3e = aCal * ARE3ae
        ARE4 = (1 - 2 * aCal) * ARF4

        # rotated IinE
        QinEe = C2e * QinE + S2e * UinE
        UinEe = C2e * UinE - S2e * QinE

        # Calibration signals and Calibration correction K from measurements with LDRCal / aCal
        if (TypeC == 2) or (TypeC == 1):  # +++++++++ rotator calibration Eq. C.4
            AT = ATE1o * IinE + (ATE4o + aCal * ATE4a) * h * VinE
            BT = aCal * (ATE3ae * QinEe - ATE2ae * h * UinEe)
            AR = ARE1o * IinE + (ARE4o + aCal * ARE4a) * h * VinE
            BR = aCal * (ARE3ae * QinEe - ARE2ae * h * UinEe)

            # Correction parameters for normal measurements; they are independent of LDR
            if not RotationErrorEpsilonForNormalMeasurements:
                # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
                GT = ATE1o * IinE + ATE4o * h * VinE
                GR = ARE1o * IinE + ARE4o * h * VinE
                HT = ATE2a * QinE + ATE3a * h * UinEe + ATE4a * h * VinE
                HR = ARE2a * QinE + ARE3a * h * UinEe + ARE4a * h * VinE
            else:
                GT = ATE1o * IinE + ATE4o * h * VinE
                GR = ARE1o * IinE + ARE4o * h * VinE
                HT = ATE2ae * QinE + ATE3ae * h * UinEe + ATE4a * h * VinE
                HR = ARE2ae * QinE + ARE3ae * h * UinEe + ARE4a * h * VinE

        elif (TypeC == 3) or (
            TypeC == 4
        ):  # +++++++++ linear polariser calibration Eq. C.5
            # p = +45°, m = -45°
            AT = (
                ATE1 * IinE + ZiC * CosC * (ATE2e * QinEe + ATE4 * VinE) + ATE3e * UinEe
            )
            BT = DiC * (ATE1 * UinEe + ATE3e * IinE) + ZiC * SinC * (
                ATE4 * QinEe - ATE2e * VinE
            )
            AR = (
                ARE1 * IinE + ZiC * CosC * (ARE2e * QinEe + ARE4 * VinE) + ARE3e * UinEe
            )
            BR = DiC * (ARE1 * UinEe + ARE3e * IinE) + ZiC * SinC * (
                ARE4 * QinEe - ARE2e * VinE
            )

            # Correction parameters for normal measurements; they are independent of LDR
            if not RotationErrorEpsilonForNormalMeasurements:
                # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
                GT = ATE1o * IinE + ATE4o * VinE
                GR = ARE1o * IinE + ARE4o * VinE
                HT = ATE2a * QinE + ATE3a * UinE + ATE4a * VinE
                HR = ARE2a * QinE + ARE3a * UinE + ARE4a * VinE
            else:
                D = IinE + DiC * QinEe
                A = DiC * IinE + QinEe
                B = ZiC * (CosC * UinEe + SinC * VinE)
                C = -ZiC * (SinC * UinEe - CosC * VinE)
                GT = ATE1o * D + ATE4o * C
                GR = ARE1o * D + ARE4o * C
                HT = ATE2a * A + ATE3a * B + ATE4a * C
                HR = ARE2a * A + ARE3a * B + ARE4a * C

        elif (
            TypeC == 6
        ):  # real HWP calibration +-22.5° rotated_diattenuator_X22x5deg.odt
            # p = +22.5°, m = -22.5°
            IE1e = np.array(
                [
                    IinE + sqr05 * DiC * QinEe,
                    sqr05 * DiC * IinE + (1 - 0.5 * WiC) * QinEe,
                    (1 - 0.5 * WiC) * UinEe + sqr05 * ZiC * SinC * VinE,
                    -sqr05 * ZiC * SinC * UinEe + ZiC * CosC * VinE,
                ]
            )
            IE2e = np.array(
                [
                    sqr05 * DiC * UinEe,
                    0.5 * WiC * UinEe - sqr05 * ZiC * SinC * VinE,
                    sqr05 * DiC * IinE + 0.5 * WiC * QinEe,
                    sqr05 * ZiC * SinC * QinEe,
                ]
            )
            ATEe = np.array([ATE1, ATE2e, ATE3e, ATE4])
            AREe = np.array([ARE1, ARE2e, ARE3e, ARE4])
            AT = np.dot(ATEe, IE1e)
            AR = np.dot(AREe, IE1e)
            BT = np.dot(ATEe, IE2e)
            BR = np.dot(AREe, IE2e)

            # Correction parameters for normal measurements; they are independent of LDR
            if not RotationErrorEpsilonForNormalMeasurements:  # calibrator taken out
                GT = ATE1o * IinE + ATE4o * VinE
                GR = ARE1o * IinE + ARE4o * VinE
                HT = ATE2a * QinE + ATE3a * UinE + ATE4a * VinE
                HR = ARE2a * QinE + ARE3a * UinE + ARE4a * VinE
            else:
                D = IinE + DiC * QinEe
                A = DiC * IinE + QinEe
                B = ZiC * (CosC * UinEe + SinC * VinE)
                C = -ZiC * (SinC * UinEe - CosC * VinE)
                GT = ATE1o * D + ATE4o * C
                GR = ARE1o * D + ARE4o * C
                HT = ATE2a * A + ATE3a * B + ATE4a * C
                HR = ARE2a * A + ARE3a * B + ARE4a * C

        else:
            print("Calibrator not implemented yet")
            sys.exit()

    else:
        print("Calibrator location not implemented yet")
        sys.exit()

    # Determination of the correction K of the calibration factor.
    IoutTp = TTa * TiC * TiO * TiE * (AT + BT)
    IoutTm = TTa * TiC * TiO * TiE * (AT - BT)
    IoutRp = TRa * TiC * TiO * TiE * (AR + BR)
    IoutRm = TRa * TiC * TiO * TiE * (AR - BR)
    # --- Results and Corrections; electronic etaR and etaT are assumed to be 1
    Etapx = IoutRp / IoutTp
    Etamx = IoutRm / IoutTm
    Etax = (Etapx * Etamx) ** 0.5

    Eta = (
        TRa / TTa
    )  # = TRa / TTa; Eta = Eta*/K  Eq. 84 => K = Eta* / Eta; equation corrected according to the papers supplement Eqs. (S.10.10.1) ff
    K = Etax / Eta

    #  For comparison with Volkers Libreoffice Müller Matrix spreadsheet
    # Eta_test_p = (IoutRp/IoutTp)
    # Eta_test_m = (IoutRm/IoutTm)
    # Eta_test = (Eta_test_p*Eta_test_m)**0.5

    # ----- random error calculation ----------
    # noise must be calculated with the photon counts of measured signals;
    # relative standard deviation of calibration signals with LDRcal; assumed to be statisitcally independent
    # normalised noise errors
    if CalcFrom0deg:
        dIoutTp = (NCalT * IoutTp) ** -0.5
        dIoutTm = (NCalT * IoutTm) ** -0.5
        dIoutRp = (NCalR * IoutRp) ** -0.5
        dIoutRm = (NCalR * IoutRm) ** -0.5
    else:
        dIoutTp = NCalT**-0.5
        dIoutTm = NCalT**-0.5
        dIoutRp = NCalR**-0.5
        dIoutRm = NCalR**-0.5
    # Forward simulated 0°-signals with LDRCal with atrue; from input file

    It = TTa * TiO * TiE * (GT + atrue * HT)
    Ir = TRa * TiO * TiE * (GR + atrue * HR)
    # relative standard deviation of standard signals with LDRmeas; assumed to be statisitcally independent
    if CalcFrom0deg:  # this works!
        dIt = (It * NI * eFacT) ** -0.5
        dIr = (Ir * NI * eFacR) ** -0.5
        """
        dIt = ((NCalT * It / IoutTp * NILfac / TCalT) ** -0.5)
        dIr = ((NCalR * Ir / IoutRp * NILfac / TCalR) ** -0.5)
        """
    else:  # does this work? Why not as above?
        dIt = (NCalT * 2 * NILfac / TCalT) ** -0.5
        dIr = (NCalR * 2 * NILfac / TCalR) ** -0.5

        # ----- Forward simulated LDRsim = 1/Eta*Ir/It  # simulated LDR* with Y from input file
    LDRsim = Ir / It  # simulated uncorrected LDR with Y from input file
    # Corrected LDRsimCorr from forward simulated LDRsim (atrue)
    # LDRsimCorr = (1./Eta*LDRsim*(GT+HT)-(GR+HR))/((GR-HR)-1./Eta*LDRsim*(GT-HT))
    """
    if ((Y == -1.) and (abs(RotL0) < 45)) or ((Y == +1.) and (abs(RotL0) > 45)):
        LDRsimx = 1. / LDRsim / Etax
    else:
        LDRsimx = LDRsim / Etax
    """
    LDRsimx = LDRsim

    # The following is correct without doubt
    # LDRCorr = (LDRsim/(Etax/K)*(GT+HT)-(GR+HR))/((GR-HR)-LDRsim/(Etax/K)*(GT-HT))

    # The following is a test whether the equations for calibration Etax and normal  signal (GHK, LDRsim) are consistent
    LDRCorr = (LDRsim / (Etax / K) * (GT + HT) - (GR + HR)) / (
        (GR - HR) - LDRsim / (Etax / K) * (GT - HT)
    )
    # here we could also use Eta instead of Etax / K => how to test whether Etax is correct? => comparison with MüllerMatrix simulation!
    # Without any correction: only measured It, Ir, EtaX are used
    LDRunCorr = LDRsim / Etax
    # LDRunCorr = (LDRsim / Etax * (GT / abs(GT) + HT / abs(HT)) - (GR / abs(GR) + HR / abs(HR))) / ((GR / abs(GR) - HR / abs(HR)) - LDRsim / Etax * (GT / abs(GT) - HT / abs(HT)))

    # LDRCorr = LDRsimx  # for test only

    F11sim = (
        1 / (TiO * TiE) * ((HR * Eta * It - HT * Ir) / (HR * GT - HT * GR))
    )  # IL = 1, Etat = Etar = 1  ;  AMT Eq.64; what is Etax/K? => see about 20 lines above: = Eta

    return (
        IoutTp,
        IoutTm,
        IoutRp,
        IoutRm,
        It,
        Ir,
        dIoutTp,
        dIoutTm,
        dIoutRp,
        dIoutRm,
        dIt,
        dIr,
        GT,
        HT,
        GR,
        HR,
        K,
        Eta,
        LDRsimx,
        LDRCorr,
        DTa,
        DRa,
        TTa,
        TRa,
        F11sim,
        LDRunCorr,
    )


# *******************************************************************************************************************************

# --- CALC with assumed true parameters from the input file
LDRtrue = LDRtrue2
(
    IoutTp0,
    IoutTm0,
    IoutRp0,
    IoutRm0,
    It0,
    Ir0,
    dIoutTp0,
    dIoutTm0,
    dIoutRp0,
    dIoutRm0,
    dIt0,
    dIr0,
    GT0,
    HT0,
    GR0,
    HR0,
    K0,
    Eta0,
    LDRsimx,
    LDRCorr,
    DTa0,
    DRa0,
    TTa0,
    TRa0,
    F11sim0,
    LDRunCorr,
) = Calc(
    TCalT,
    TCalR,
    NCalT,
    NCalR,
    Qin0,
    Vin0,
    RotL0,
    RotE0,
    RetE0,
    DiE0,
    RotO0,
    RetO0,
    DiO0,
    RotC0,
    RetC0,
    DiC0,
    TP0,
    TS0,
    RP0,
    RS0,
    ERaT0,
    RotaT0,
    RetT0,
    ERaR0,
    RotaR0,
    RetR0,
    LDRCal0,
)
Etax0 = K0 * Eta0
Etapx0 = IoutRp0 / IoutTp0
Etamx0 = IoutRm0 / IoutTm0
# --- Print parameters to console and output file
FileName = "output_" + InputPath.stem + "_" + fname[0:-3] + ".dat"
output_path = os.path.join(OutputDir, FileName)
# output_path = OutputFile
# with open('output_files\\' + OutputFile, 'w') as f:
with open(output_path, "w") as f:
    with redirect_stdout(f):
        print("From ", dname)
        print("Running ", fname)
        print(
            "Reading input file ", InputPath
        )  # , "  for Lidar system :", EID, ", ", LID)
        print("for Lidar system: ", EID, ", ", LID)
        # --- Print iput information*********************************
        print(" --- Input parameters: value ±error / ±steps  ----------------------")
        print(
            "{0:7}{1:17} {2:6.4f}±{3:7.4f}/{4:2d}".format(
                "Laser: ", "Qin =", Qin0, dQin, nQin
            )
        )
        print(
            "{0:7}{1:17} {2:6.4f}±{3:7.4f}/{4:2d}".format("", "Vin =", Vin0, dVin, nVin)
        )
        print(
            "{0:7}{1:17} {2:6.4f}±{3:7.4f}/{4:2d}".format(
                "", "Rotation alpha = ", RotL0, dRotL, nRotL
            )
        )
        print(
            "{0:7}{1:15} {2:8.4f} {3:17}".format(
                "",
                "=> DOP",
                ((Qin**2 + Vin**2) ** 0.5),
                " (degree of polarisation)",
            )
        )

        print(
            "Optic:        Diatt.,                 Tunpol,   Retard.,   Rotation (deg)"
        )
        print(
            "{0:12} {1:7.4f}  ±{2:7.4f}  /{8:2d}, {3:7.4f}, {4:3.0f}±{5:3.0f}/{9:2d}, {6:7.4f}±{7:7.4f}/{10:2d}".format(
                "Emitter    ",
                DiE0,
                dDiE,
                TiE,
                RetE0,
                dRetE,
                RotE0,
                dRotE,
                nDiE,
                nRetE,
                nRotE,
            )
        )
        print(
            "{0:12} {1:7.4f}  ±{2:7.4f}  /{8:2d}, {3:7.4f}, {4:3.0f}±{5:3.0f}/{9:2d}, {6:7.4f}±{7:7.4f}/{10:2d}".format(
                "Receiver   ",
                DiO0,
                dDiO,
                TiO,
                RetO0,
                dRetO,
                RotO0,
                dRotO,
                nDiO,
                nRetO,
                nRotO,
            )
        )
        print(
            "{0:12} {1:9.6f}±{2:9.6f}/{8:2d}, {3:7.4f}, {4:3.0f}±{5:3.0f}/{9:2d}, {6:7.4f}±{7:7.4f}/{10:2d}".format(
                "Calibrator ",
                DiC0,
                dDiC,
                TiC,
                RetC0,
                dRetC,
                RotC0,
                dRotC,
                nDiC,
                nRetC,
                nRotC,
            )
        )
        print("{0:12}".format(" Pol.-filter ------ "))
        print(
            "{0:12}{1:7.4f}±{2:7.4f}/{3:2d}, {4:7.4f}±{5:7.4f}/{6:2d}".format(
                "ERT, RotT       :", ERaT0, dERaT, nERaT, RotaT0, dRotaT, nRotaT
            )
        )
        print(
            "{0:12}{1:7.4f}±{2:7.4f}/{3:2d}, {4:7.4f}±{5:7.4f}/{6:2d}".format(
                "ERR, RotR       :", ERaR0, dERaR, nERaR, RotaR0, dRotaR, nRotaR
            )
        )
        print("{0:12}".format(" PBS ------ "))
        print(
            "{0:12}{1:7.4f}±{2:7.4f}/{3:2d}, {4:7.4f}±{5:7.4f}/{6:2d}".format(
                "TP,TS           :", TP0, dTP, nTP, TS0, dTS, nTS
            )
        )
        print(
            "{0:12}{1:7.4f}±{2:7.4f}/{3:2d}, {4:7.4f}±{5:7.4f}/{6:2d}".format(
                "RP,RS           :", RP0, dRP, nRP, RS0, dRS, nRS
            )
        )
        print(
            "{0:12}{1:7.4f},{2:7.4f}, {3:7.4f},{4:7.4f}, {5:1.0f}".format(
                "DT,TT,DR,TR,Y   :", DiT, TiT, DiR, TiR, Y
            )
        )
        print("{0:12}".format(" Combined PBS + Pol.-filter ------ "))
        print(
            "{0:12}{1:7.4f},{2:7.4f}, {3:7.4f},{4:7.4f}".format(
                "DT,TT,DR,TR     :", DTa0, TTa0, DRa0, TRa0
            )
        )
        print(
            "{0:26}: {1:6.3f}± {2:5.3f}/{3:2d}".format(
                "LDRCal during calibration in calibration range",
                LDRCal0,
                dLDRCal,
                nLDRCal,
            )
        )
        print(
            "{0:12}".format(
                " --- Additional ND filter attenuation (transmission) during the calibration ---"
            )
        )
        print(
            "{0:12}{1:7.4f}±{2:7.4f}/{3:2d}, {4:7.4f}±{5:7.4f}/{6:2d}".format(
                "TCalT,TCalR      :", TCalT0, dTCalT, nTCalT, TCalR0, dTCalR, nTCalR
            )
        )
        print()
        print(Type[TypeC], "calibrator is located", Loc[LocC])
        print(
            "Rotation error epsilon is considered also for normal measurements = ",
            RotationErrorEpsilonForNormalMeasurements,
        )
        print("The PBS incidence plane is ", dY[int(Y + 1)], "to the reference plane")
        print(
            "The laser polarisation in the reference plane is finally",
            dY2[int(Y + 1)],
            "=>",
            dY3,
        )
        print("RS_RP_depend_on_TS_TP = ", RS_RP_depend_on_TS_TP)
        #  end of print actual system parameters
        # ******************************************************************************

        print()

        K0List = np.zeros(7)
        LDRsimxList = np.zeros(7)
        LDRCalList = LDRCal0, 0.004, 0.05, 0.1, 0.2, 0.3, 0.45
        # The loop over LDRCalList is ony for checking whether and how much the LDR depends on the LDRCal during calibration and whether the corrections work.
        # Still with assumed true parameters in input file

        """
        facIt = NCalT / TCalT0 * NILfac
        facIr = NCalR / TCalR0 * NILfac
        """
        facIt = NI * eFacT
        facIr = NI * eFacR
        if bPlotEtax:
            # check error signals
            # dIs are relative stdevs
            print(
                "LDRCal, IoutTp,   IoutTm,     IoutRp,        IoutRm,         It,          Ir,      dIoutTp,dIoutTm,dIoutRp,dIoutRm,dIt,   dIr"
            )

        for i, LDRCal in enumerate(LDRCalList):
            (
                IoutTp,
                IoutTm,
                IoutRp,
                IoutRm,
                It,
                Ir,
                dIoutTp,
                dIoutTm,
                dIoutRp,
                dIoutRm,
                dIt,
                dIr,
                GT0,
                HT0,
                GR0,
                HR0,
                K0,
                Eta0,
                LDRsimx,
                LDRCorr,
                DTa0,
                DRa0,
                TTa0,
                TRa0,
                F11sim0,
                LDRunCorr,
            ) = Calc(
                TCalT0,
                TCalR0,
                NCalT,
                NCalR,
                Qin0,
                Vin0,
                RotL0,
                RotE0,
                RetE0,
                DiE0,
                RotO0,
                RetO0,
                DiO0,
                RotC0,
                RetC0,
                DiC0,
                TP0,
                TS0,
                RP0,
                RS0,
                ERaT0,
                RotaT0,
                RetT0,
                ERaR0,
                RotaR0,
                RetR0,
                LDRCal,
            )
            K0List[i] = K0
            LDRsimxList[i] = LDRsimx

            if bPlotEtax:
                # check error signals
                print(
                    "{:0.2f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}".format(
                        LDRCal,
                        IoutTp * NCalT,
                        IoutTm * NCalT,
                        IoutRp * NCalR,
                        IoutRm * NCalR,
                        It * facIt,
                        Ir * facIr,
                        dIoutTp,
                        dIoutTm,
                        dIoutRp,
                        dIoutRm,
                        dIt,
                        dIr,
                    )
                )
                # print( "{:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}".format(IoutTp, IoutTm, IoutRp, IoutRm, It, Ir, dIoutTp, dIoutTm, dIoutRp, dIoutRm, dIt, dIr))
                # end check error signals
        print(
            "==========================================================================================================="
        )
        print(
            "{0:8},{1:8},{2:8},{3:8},{4:4}{5:5.3f}{6:1},{7:8},{8:9},{9:9},{10:9},{11:9},{12:9}".format(
                " GR",
                " GT",
                " HR",
                " HT",
                "  K(",
                LDRCalList[0],
                ")",
                "  K(0.004)",
                " K(0.05)",
                "  K(0.1)",
                "  K(0.2)",
                "  K(0.3)",
                "  K(0.45)",
            )
        )
        print(
            "{0:8.5f},{1:8.5f},{2:8.5f},{3:8.5f},{4:9.5f} ,{5:9.5f},{6:9.5f},{7:10.5f},{8:9.5f},{9:9.5f},{10:9.5f}".format(
                GR0,
                GT0,
                HR0,
                HT0,
                K0List[0],
                K0List[1],
                K0List[2],
                K0List[3],
                K0List[4],
                K0List[5],
                K0List[6],
            )
        )
        print(
            "==========================================================================================================="
        )
        print()
        # --- Calculate 2nd order polynom-fit to K(LDRCalList)
        K_poly = np.polyfit(LDRCalList, K0List, 2)
        # p = np.poly1d(K_poly)
        # f-string formatting
        print("2nd order polyfit: K(LDRcal) = a + (b * LDRcal) + (c * LDRcal^2)")
        print(
            "(LDRcal is the volume linear depolarisation ratio in the range used for the polarisation calibration)"
        )
        print(" a = ", f"{K_poly[2]:.5f}")
        print(" b = ", f"{K_poly[1]:.5f}")
        print(" c = ", f"{K_poly[0]:.5f}")
        print()

        print("Errors from neglecting GHK corrections and/or calibration:")
        print(
            "{0:>10},{1:>10},{2:>10},{3:>10},{4:>10},{5:>10}".format(
                "LDRtrue", "LDRunCorr", "1/LDRunCorr", "LDRsimx", "1/LDRsimx", "LDRCorr"
            )
        )

        aF11sim0 = np.zeros(5)
        LDRrange = np.zeros(5)
        LDRsim0 = np.zeros(5)
        LDRrange = [0.004, 0.02, 0.1, 0.3, 0.45]  # list
        LDRrange[0] = LDRtrue2  # value in the input file; default 0.004

        # The loop over LDRtrueList is only for checking how much the uncorrected LDRsimx deviates from LDRtrue ... and whether the corrections work.
        # LDRsimx = LDRsim = Ir / It    or      1/LDRsim
        # Still with assumed true parameters in input file
        for i, LDRtrue in enumerate(LDRrange):
            # for LDRtrue in LDRrange:
            (
                IoutTp,
                IoutTm,
                IoutRp,
                IoutRm,
                It,
                Ir,
                dIoutTp,
                dIoutTm,
                dIoutRp,
                dIoutRm,
                dIt,
                dIr,
                GT0,
                HT0,
                GR0,
                HR0,
                K0,
                Eta0,
                LDRsimx,
                LDRCorr,
                DTa0,
                DRa0,
                TTa0,
                TRa0,
                F11sim0,
                LDRunCorr,
            ) = Calc(
                TCalT0,
                TCalR0,
                NCalT,
                NCalR,
                Qin0,
                Vin0,
                RotL0,
                RotE0,
                RetE0,
                DiE0,
                RotO0,
                RetO0,
                DiO0,
                RotC0,
                RetC0,
                DiC0,
                TP0,
                TS0,
                RP0,
                RS0,
                ERaT0,
                RotaT0,
                RetT0,
                ERaR0,
                RotaR0,
                RetR0,
                LDRCal0,
            )
            print(
                "{0:10.5f},{1:10.5f},{2:10.5f},{3:10.5f},{4:10.5f},{5:10.5f}".format(
                    LDRtrue, LDRunCorr, 1 / LDRunCorr, LDRsimx, 1 / LDRsimx, LDRCorr
                )
            )
            aF11sim0[i] = F11sim0
            LDRsim0[i] = Ir / It
            # the assumed true aF11sim0 results will be used below to calc the deviation from the real signals
        print(
            "LDRsimx = LDR of the nominal system directly from measured signals without  calibration and GHK-corrections"
        )
        print(
            "LDRunCorr = LDR of the nominal system directly from measured signals with calibration but without  GHK-corrections; electronic amplifications = 1 assumed"
        )
        print("LDRCorr = LDR calibrated and GHK-corrected")
        print()
        print("Errors from signal noise:")
        print(
            "Signal counts: NI, NCalT, NCalR, NILfac, nNCal, nNI, stdev(NI)/NI = {0:10.0f},{1:10.0f},{2:10.0f},{3:3.0f},{4:2.0f},{5:2.0f},{6:8.5f}".format(
                NI, NCalT, NCalR, NILfac, nNCal, nNI, 1.0 / NI**0.5
            )
        )
        print()
        print()
        """# das muß wieder weg
        print("IoutTp, IoutTm, IoutRp, IoutRm, It    , Ir    , dIoutTp, dIoutTm, dIoutRp, dIoutRm, dIt, dIr")
        LDRCal = 0.01
        for i, LDRtrue in enumerate(LDRrange):
            IoutTp, IoutTm, IoutRp, IoutRm, It, Ir, dIoutTp, dIoutTm, dIoutRp, dIoutRm, dIt, dIr, \
            GT0, HT0, GR0, HR0, K0, Eta0, LDRsimx, LDRCorr, DTa0, DRa0, TTa0, TRa0, F11sim0, LDRunCorr = \
            Calc(TCalT0, TCalR0, NCalT, NCalR, DOLP0, RotL0, RotE0, RetE0, DiE0,
                 RotO0, RetO0, DiO0, RotC0, RetC0, DiC0, TP0, TS0, RP0, RS0,
                 ERaT0, RotaT0, RetT0, ERaR0, RotaR0, RetR0, LDRCal0)
            print( "{:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}".format(
                IoutTp * NCalT, IoutTm * NCalT, IoutRp * NCalR, IoutRm * NCalR, It * facIt, Ir * facIr,
                dIoutTp, dIoutTm, dIoutRp, dIoutRm, dIt, dIr))
            aF11sim0[i] = F11sim0
            # the assumed true aF11sim0 results will be used below to calc the deviation from the real signals
        # bis hierher weg
        """

# file = open('output_files\\' + OutputFile, 'r')
file = open(output_path, "r")
print(file.read())
file.close()

# --- CALC again assumed truth with LDRCal0 and with assumed true parameters in input file to reset all 0-values
LDRtrue = LDRtrue2
(
    IoutTp0,
    IoutTm0,
    IoutRp0,
    IoutRm0,
    It0,
    Ir0,
    dIoutTp0,
    dIoutTm0,
    dIoutRp0,
    dIoutRm0,
    dIt0,
    dIr0,
    GT0,
    HT0,
    GR0,
    HR0,
    K0,
    Eta0,
    LDRsimx,
    LDRCorr,
    DTa0,
    DRa0,
    TTa0,
    TRa0,
    F11sim0,
    LDRunCorr,
) = Calc(
    TCalT0,
    TCalR0,
    NCalT,
    NCalR,
    Qin0,
    Vin0,
    RotL0,
    RotE0,
    RetE0,
    DiE0,
    RotO0,
    RetO0,
    DiO0,
    RotC0,
    RetC0,
    DiC0,
    TP0,
    TS0,
    RP0,
    RS0,
    ERaT0,
    RotaT0,
    RetT0,
    ERaR0,
    RotaR0,
    RetR0,
    LDRCal0,
)
Etax0 = K0 * Eta0
Etapx0 = IoutRp0 / IoutTp0
Etamx0 = IoutRm0 / IoutTm0
"""
if(PrintToOutputFile):
    f = open('output_ver7.dat', 'w')
    old_target = sys.stdout
    sys.stdout = f

    print("something")

if(PrintToOutputFile):
    sys.stdout.flush()
    f.close
    sys.stdout = old_target
"""
if Error_Calc:
    print("-----------------------------------------")
    print("Errors from GHK correction uncertainties:")
    print("-----------------------------------------")
    # --- CALC again assumed truth with LDRCal0 and with assumed true parameters in input file to reset all 0-values
    LDRtrue = LDRtrue2
    (
        IoutTp0,
        IoutTm0,
        IoutRp0,
        IoutRm0,
        It0,
        Ir0,
        dIoutTp0,
        dIoutTm0,
        dIoutRp0,
        dIoutRm0,
        dIt0,
        dIr0,
        GT0,
        HT0,
        GR0,
        HR0,
        K0,
        Eta0,
        LDRsimx,
        LDRCorr,
        DTa0,
        DRa0,
        TTa0,
        TRa0,
        F11sim0,
        LDRunCorr,
    ) = Calc(
        TCalT0,
        TCalR0,
        NCalT,
        NCalR,
        Qin0,
        Vin0,
        RotL0,
        RotE0,
        RetE0,
        DiE0,
        RotO0,
        RetO0,
        DiO0,
        RotC0,
        RetC0,
        DiC0,
        TP0,
        TS0,
        RP0,
        RS0,
        ERaT0,
        RotaT0,
        RetT0,
        ERaR0,
        RotaR0,
        RetR0,
        LDRCal0,
    )
    Etax0 = K0 * Eta0
    Etapx0 = IoutRp0 / IoutTp0
    Etamx0 = IoutRm0 / IoutTm0

    # --- Start Error calculation with variable parameters ------------------------------------------------------------------
    # error nNCal: one-sigma in steps to left and right for calibration signals
    # error nNI: one-sigma in steps to left and right for 0° signals

    iN = -1
    N = (
        (nTCalT * 2 + 1)
        * (nTCalR * 2 + 1)
        * (nNCal * 2 + 1) ** 4
        * (nNI * 2 + 1) ** 2
        * (nQin * 2 + 1)
        * (nVin * 2 + 1)
        * (nRotL * 2 + 1)
        * (nRotE * 2 + 1)
        * (nRetE * 2 + 1)
        * (nDiE * 2 + 1)
        * (nRotO * 2 + 1)
        * (nRetO * 2 + 1)
        * (nDiO * 2 + 1)
        * (nRotC * 2 + 1)
        * (nRetC * 2 + 1)
        * (nDiC * 2 + 1)
        * (nTP * 2 + 1)
        * (nTS * 2 + 1)
        * (nRP * 2 + 1)
        * (nRS * 2 + 1)
        * (nERaT * 2 + 1)
        * (nERaR * 2 + 1)
        * (nRotaT * 2 + 1)
        * (nRotaR * 2 + 1)
        * (nRetT * 2 + 1)
        * (nRetR * 2 + 1)
        * (nLDRCal * 2 + 1)
    )
    print("number of system variations N = ", N, " ", end="")

    if N > 1e6:
        if (
            user_yes_no_query(
                "Warning: processing "
                + str(N)
                + " samples will take very long. Do you want to proceed?"
            )
            == 0
        ):
            sys.exit()
    if N > 5e6:
        if (
            user_yes_no_query(
                "Warning: the memory required for "
                + str(N)
                + " samples might be "
                + "{0:5.1f}".format(N / 4e6)
                + " GB. Do you anyway want to proceed?"
            )
            == 0
        ):
            sys.exit()

    # if user_yes_no_query('Warning: processing' + str(N) + ' samples will take very long. Do you want to proceed?') == 0: sys.exit()

    # --- Arrays for plotting ------
    LDRmin = np.zeros(5)
    LDRmax = np.zeros(5)
    LDRstd = np.zeros(5)
    LDRmean = np.zeros(5)
    LDRmedian = np.zeros(5)
    LDRskew = np.zeros(5)
    LDRkurt = np.zeros(5)
    LDRsimmin = np.zeros(5)
    LDRsimmax = np.zeros(5)
    LDRsimmean = np.zeros(5)

    F11min = np.zeros(5)
    F11max = np.zeros(5)
    Etaxmin = np.zeros(5)
    Etaxmax = np.zeros(5)

    aQin = np.zeros(N)
    aVin = np.zeros(N)
    aERaT = np.zeros(N)
    aERaR = np.zeros(N)
    aRotaT = np.zeros(N)
    aRotaR = np.zeros(N)
    aRetT = np.zeros(N)
    aRetR = np.zeros(N)
    aTP = np.zeros(N)
    aTS = np.zeros(N)
    aRP = np.zeros(N)
    aRS = np.zeros(N)
    aDiE = np.zeros(N)
    aDiO = np.zeros(N)
    aDiC = np.zeros(N)
    aRotC = np.zeros(N)
    aRetC = np.zeros(N)
    aRotL = np.zeros(N)
    aRetE = np.zeros(N)
    aRotE = np.zeros(N)
    aRetO = np.zeros(N)
    aRotO = np.zeros(N)
    aLDRCal = np.zeros(N)
    aNCalTp = np.zeros(N)
    aNCalTm = np.zeros(N)
    aNCalRp = np.zeros(N)
    aNCalRm = np.zeros(N)
    aNIt = np.zeros(N)
    aNIr = np.zeros(N)
    aTCalT = np.zeros(N)
    aTCalR = np.zeros(N)

    # each np.zeros((LDRrange, N)) array has the same N-dependency
    aLDRcorr = np.zeros((5, N))
    aLDRsim = np.zeros((5, N))
    aF11corr = np.zeros((5, N))
    aPLDR = np.zeros((5, N))
    aEtax = np.zeros((5, N))
    aEtapx = np.zeros((5, N))
    aEtamx = np.zeros((5, N))

    # np.zeros((GHKs, N))
    aGHK = np.zeros((5, N))

    atime = clock()
    dtime = clock()

    # --- Calc Error signals
    # ---- Do the calculations of bra-ket vectors
    h = -1.0 if TypeC == 2 else 1

    for iLDRCal in range(-nLDRCal, nLDRCal + 1):
        # from input file:  LDRCal for calibration measurements
        LDRCal = LDRCal0
        if nLDRCal > 0:
            LDRCal = LDRCal0 + iLDRCal * dLDRCal / nLDRCal
            # provides the intensities of the calibration measurements at various LDRCal for signal noise errors
            # IoutTp, IoutTm, IoutRp, IoutRm, dIoutTp, dIoutTm, dIoutRp, dIoutRm

        aCal = (1.0 - LDRCal) / (1.0 + LDRCal)
        for iQin, iVin, iRotL, iRotE, iRetE, iDiE in [
            (iQin, iVin, iRotL, iRotE, iRetE, iDiE)
            for iQin in range(-nQin, nQin + 1)
            for iVin in range(-nVin, nVin + 1)
            for iRotL in range(-nRotL, nRotL + 1)
            for iRotE in range(-nRotE, nRotE + 1)
            for iRetE in range(-nRetE, nRetE + 1)
            for iDiE in range(-nDiE, nDiE + 1)
        ]:

            if nQin > 0:
                Qin = Qin0 + iQin * dQin / nQin
            if nVin > 0:
                Vin = Vin0 + iVin * dVin / nVin
            if nRotL > 0:
                RotL = RotL0 + iRotL * dRotL / nRotL
            if nRotE > 0:
                RotE = RotE0 + iRotE * dRotE / nRotE
            if nRetE > 0:
                RetE = RetE0 + iRetE * dRetE / nRetE
            if nDiE > 0:
                DiE = DiE0 + iDiE * dDiE / nDiE

            if ((Qin**2 + Vin**2) ** 0.5) > 1.0:
                print("Error: degree of polarisation of laser > 1. Check Qin and Vin! ")
                sys.exit()
            # angles of emitter and laser and calibrator and receiver optics
            # RotL = alpha, RotE = beta, RotO = gamma, RotC = epsilon
            S2a = np.sin(2 * np.deg2rad(RotL))
            C2a = np.cos(2 * np.deg2rad(RotL))
            S2b = np.sin(2 * np.deg2rad(RotE))
            C2b = np.cos(2 * np.deg2rad(RotE))
            S2ab = np.sin(np.deg2rad(2 * RotL - 2 * RotE))
            C2ab = np.cos(np.deg2rad(2 * RotL - 2 * RotE))

            # Laser with Degree of linear polarization DOLP
            IinL = 1.0
            QinL = Qin
            UinL = 0.0
            VinL = Vin
            # VinL = (1. - DOLP ** 2) ** 0.5

            # Stokes Input Vector rotation Eq. E.4
            A = C2a * QinL - S2a * UinL
            B = S2a * QinL + C2a * UinL
            # Stokes Input Vector rotation Eq. E.9
            C = C2ab * QinL - S2ab * UinL
            D = S2ab * QinL + C2ab * UinL

            # emitter optics
            CosE = np.cos(np.deg2rad(RetE))
            SinE = np.sin(np.deg2rad(RetE))
            ZiE = (1.0 - DiE**2) ** 0.5
            WiE = 1.0 - ZiE * CosE

            # Stokes Input Vector after emitter optics equivalent to Eq. E.9 with already rotated input vector from Eq. E.4
            # b = beta
            IinE = IinL + DiE * C
            QinE = C2b * DiE * IinL + A + S2b * (WiE * D - ZiE * SinE * VinL)
            UinE = S2b * DiE * IinL + B - C2b * (WiE * D - ZiE * SinE * VinL)
            VinE = -ZiE * SinE * D + ZiE * CosE * VinL

            # -------------------------
            # F11 assuemd to be = 1  => measured: F11m = IinP / IinE with atrue
            # F11sim = (IinE + DiO*atrue*(C2g*QinE - S2g*UinE))/IinE
            # -------------------------

            for (
                iRotO,
                iRetO,
                iDiO,
                iRotC,
                iRetC,
                iDiC,
                iTP,
                iTS,
                iRP,
                iRS,
                iERaT,
                iRotaT,
                iRetT,
                iERaR,
                iRotaR,
                iRetR,
            ) in [
                (
                    iRotO,
                    iRetO,
                    iDiO,
                    iRotC,
                    iRetC,
                    iDiC,
                    iTP,
                    iTS,
                    iRP,
                    iRS,
                    iERaT,
                    iRotaT,
                    iRetT,
                    iERaR,
                    iRotaR,
                    iRetR,
                )
                for iRotO in range(-nRotO, nRotO + 1)
                for iRetO in range(-nRetO, nRetO + 1)
                for iDiO in range(-nDiO, nDiO + 1)
                for iRotC in range(-nRotC, nRotC + 1)
                for iRetC in range(-nRetC, nRetC + 1)
                for iDiC in range(-nDiC, nDiC + 1)
                for iTP in range(-nTP, nTP + 1)
                for iTS in range(-nTS, nTS + 1)
                for iRP in range(-nRP, nRP + 1)
                for iRS in range(-nRS, nRS + 1)
                for iERaT in range(-nERaT, nERaT + 1)
                for iRotaT in range(-nRotaT, nRotaT + 1)
                for iRetT in range(-nRetT, nRetT + 1)
                for iERaR in range(-nERaR, nERaR + 1)
                for iRotaR in range(-nRotaR, nRotaR + 1)
                for iRetR in range(-nRetR, nRetR + 1)
            ]:

                if nRotO > 0:
                    RotO = RotO0 + iRotO * dRotO / nRotO
                if nRetO > 0:
                    RetO = RetO0 + iRetO * dRetO / nRetO
                if nDiO > 0:
                    DiO = DiO0 + iDiO * dDiO / nDiO
                if nRotC > 0:
                    RotC = RotC0 + iRotC * dRotC / nRotC
                if nRetC > 0:
                    RetC = RetC0 + iRetC * dRetC / nRetC
                if nDiC > 0:
                    DiC = DiC0 + iDiC * dDiC / nDiC
                if nTP > 0:
                    TP = TP0 + iTP * dTP / nTP
                if nTS > 0:
                    TS = TS0 + iTS * dTS / nTS
                if nRP > 0:
                    RP = RP0 + iRP * dRP / nRP
                if nRS > 0:
                    RS = RS0 + iRS * dRS / nRS
                if nERaT > 0:
                    ERaT = ERaT0 + iERaT * dERaT / nERaT
                if nRotaT > 0:
                    RotaT = RotaT0 + iRotaT * dRotaT / nRotaT
                if nRetT > 0:
                    RetT = RetT0 + iRetT * dRetT / nRetT
                if nERaR > 0:
                    ERaR = ERaR0 + iERaR * dERaR / nERaR
                if nRotaR > 0:
                    RotaR = RotaR0 + iRotaR * dRotaR / nRotaR
                if nRetR > 0:
                    RetR = RetR0 + iRetR * dRetR / nRetR

                # print("{0:5.2f}, {1:5.2f}, {2:5.2f}, {3:10d}".format(RotL, RotE, RotO, iN))

                # receiver optics
                CosO = np.cos(np.deg2rad(RetO))
                SinO = np.sin(np.deg2rad(RetO))
                ZiO = (1.0 - DiO**2) ** 0.5
                WiO = 1.0 - ZiO * CosO
                S2g = np.sin(np.deg2rad(2 * RotO))
                C2g = np.cos(np.deg2rad(2 * RotO))
                # calibrator
                CosC = np.cos(np.deg2rad(RetC))
                SinC = np.sin(np.deg2rad(RetC))
                ZiC = (1.0 - DiC**2) ** 0.5
                WiC = 1.0 - ZiC * CosC

                # analyser
                # For POLLY_XTs
                if RS_RP_depend_on_TS_TP:
                    RS = 1.0 - TS
                    RP = 1.0 - TP
                TiT = 0.5 * (TP + TS)
                DiT = (TP - TS) / (TP + TS)
                ZiT = (1.0 - DiT**2.0) ** 0.5
                TiR = 0.5 * (RP + RS)
                DiR = (RP - RS) / (RP + RS)
                ZiR = (1.0 - DiR**2.0) ** 0.5
                CosT = np.cos(np.deg2rad(RetT))
                SinT = np.sin(np.deg2rad(RetT))
                CosR = np.cos(np.deg2rad(RetR))
                SinR = np.sin(np.deg2rad(RetR))

                # cleaning pol-filter
                DaT = (1.0 - ERaT) / (1.0 + ERaT)
                DaR = (1.0 - ERaR) / (1.0 + ERaR)
                TaT = 0.5 * (1.0 + ERaT)
                TaR = 0.5 * (1.0 + ERaR)

                S2aT = np.sin(np.deg2rad(h * 2.0 * RotaT))
                C2aT = np.cos(np.deg2rad(2.0 * RotaT))
                S2aR = np.sin(np.deg2rad(h * 2.0 * RotaR))
                C2aR = np.cos(np.deg2rad(2.0 * RotaR))

                # Analyzer As before the PBS Eq. D.5; combined PBS and cleaning pol-filter
                ATPT = 1 + C2aT * DaT * DiT  # unpolarized transmission correction
                TTa = TiT * TaT * ATPT  # unpolarized transmission
                ATP1 = 1.0
                ATP2 = Y * (DiT + C2aT * DaT) / ATPT
                ATP3 = Y * S2aT * DaT * ZiT * CosT / ATPT
                ATP4 = S2aT * DaT * ZiT * SinT / ATPT
                ATP = np.array([ATP1, ATP2, ATP3, ATP4])
                DTa = ATP2 * Y

                ARPT = 1 + C2aR * DaR * DiR  # unpolarized transmission correction
                TRa = TiR * TaR * ARPT  # unpolarized transmission
                ARP1 = 1
                ARP2 = Y * (DiR + C2aR * DaR) / ARPT
                ARP3 = Y * S2aR * DaR * ZiR * CosR / ARPT
                ARP4 = S2aR * DaR * ZiR * SinR / ARPT
                ARP = np.array([ARP1, ARP2, ARP3, ARP4])
                DRa = ARP2 * Y

                # ---- Calculate signals and correction parameters for diffeent locations and calibrators
                if LocC == 4:  # Calibrator before the PBS
                    # print("Calibrator location not implemented yet")

                    # S2ge = np.sin(np.deg2rad(2*RotO + h*2*RotC))
                    # C2ge = np.cos(np.deg2rad(2*RotO + h*2*RotC))
                    S2e = np.sin(np.deg2rad(h * 2 * RotC))
                    C2e = np.cos(np.deg2rad(2 * RotC))
                    # rotated AinP by epsilon Eq. C.3
                    ATP2e = C2e * ATP2 + S2e * ATP3
                    ATP3e = C2e * ATP3 - S2e * ATP2
                    ARP2e = C2e * ARP2 + S2e * ARP3
                    ARP3e = C2e * ARP3 - S2e * ARP2
                    ATPe = np.array([ATP1, ATP2e, ATP3e, ATP4])
                    ARPe = np.array([ARP1, ARP2e, ARP3e, ARP4])
                    # Stokes Input Vector before the polarising beam splitter Eq. E.31
                    A = C2g * QinE - S2g * UinE
                    B = S2g * QinE + C2g * UinE
                    # C = (WiO*aCal*B + ZiO*SinO*(1-2*aCal)*VinE)
                    Co = ZiO * SinO * VinE
                    Ca = WiO * B - 2 * ZiO * SinO * VinE
                    # C = Co + aCal*Ca
                    # IinP = (IinE + DiO*aCal*A)
                    # QinP = (C2g*DiO*IinE + aCal*QinE - S2g*C)
                    # UinP = (S2g*DiO*IinE - aCal*UinE + C2g*C)
                    # VinP = (ZiO*SinO*aCal*B + ZiO*CosO*(1-2*aCal)*VinE)
                    IinPo = IinE
                    QinPo = C2g * DiO * IinE - S2g * Co
                    UinPo = S2g * DiO * IinE + C2g * Co
                    VinPo = ZiO * CosO * VinE

                    IinPa = DiO * A
                    QinPa = QinE - S2g * Ca
                    UinPa = -UinE + C2g * Ca
                    VinPa = ZiO * (SinO * B - 2 * CosO * VinE)

                    IinP = IinPo + aCal * IinPa
                    QinP = QinPo + aCal * QinPa
                    UinP = UinPo + aCal * UinPa
                    VinP = VinPo + aCal * VinPa
                    # Stokes Input Vector before the polarising beam splitter rotated by epsilon Eq. C.3
                    # QinPe = C2e*QinP + S2e*UinP
                    # UinPe = C2e*UinP - S2e*QinP
                    QinPoe = C2e * QinPo + S2e * UinPo
                    UinPoe = C2e * UinPo - S2e * QinPo
                    QinPae = C2e * QinPa + S2e * UinPa
                    UinPae = C2e * UinPa - S2e * QinPa
                    QinPe = C2e * QinP + S2e * UinP
                    UinPe = C2e * UinP - S2e * QinP

                    # Calibration signals and Calibration correction K from measurements with LDRCal / aCal
                    if (TypeC == 2) or (TypeC == 1):  # rotator calibration Eq. C.4
                        # parameters for calibration with aCal
                        AT = ATP1 * IinP + h * ATP4 * VinP
                        BT = ATP3e * QinP - h * ATP2e * UinP
                        AR = ARP1 * IinP + h * ARP4 * VinP
                        BR = ARP3e * QinP - h * ARP2e * UinP
                        # Correction parameters for normal measurements; they are independent of LDR
                        if (
                            not RotationErrorEpsilonForNormalMeasurements
                        ):  # calibrator taken out
                            IS1 = np.array([IinPo, QinPo, UinPo, VinPo])
                            IS2 = np.array([IinPa, QinPa, UinPa, VinPa])
                            GT = np.dot(ATP, IS1)
                            GR = np.dot(ARP, IS1)
                            HT = np.dot(ATP, IS2)
                            HR = np.dot(ARP, IS2)
                        else:
                            IS1 = np.array([IinPo, QinPo, UinPo, VinPo])
                            IS2 = np.array([IinPa, QinPa, UinPa, VinPa])
                            GT = np.dot(ATPe, IS1)
                            GR = np.dot(ARPe, IS1)
                            HT = np.dot(ATPe, IS2)
                            HR = np.dot(ARPe, IS2)
                    elif (TypeC == 3) or (
                        TypeC == 4
                    ):  # linear polariser calibration Eq. C.5
                        # parameters for calibration with aCal
                        AT = (
                            ATP1 * IinP
                            + ATP3e * UinPe
                            + ZiC * CosC * (ATP2e * QinPe + ATP4 * VinP)
                        )
                        BT = DiC * (ATP1 * UinPe + ATP3e * IinP) - ZiC * SinC * (
                            ATP2e * VinP - ATP4 * QinPe
                        )
                        AR = (
                            ARP1 * IinP
                            + ARP3e * UinPe
                            + ZiC * CosC * (ARP2e * QinPe + ARP4 * VinP)
                        )
                        BR = DiC * (ARP1 * UinPe + ARP3e * IinP) - ZiC * SinC * (
                            ARP2e * VinP - ARP4 * QinPe
                        )
                        # Correction parameters for normal measurements; they are independent of LDR
                        if (
                            not RotationErrorEpsilonForNormalMeasurements
                        ):  # calibrator taken out
                            IS1 = np.array([IinPo, QinPo, UinPo, VinPo])
                            IS2 = np.array([IinPa, QinPa, UinPa, VinPa])
                            GT = np.dot(ATP, IS1)
                            GR = np.dot(ARP, IS1)
                            HT = np.dot(ATP, IS2)
                            HR = np.dot(ARP, IS2)
                        else:
                            IS1e = np.array(
                                [
                                    IinPo + DiC * QinPoe,
                                    DiC * IinPo + QinPoe,
                                    ZiC * (CosC * UinPoe + SinC * VinPo),
                                    -ZiC * (SinC * UinPoe - CosC * VinPo),
                                ]
                            )
                            IS2e = np.array(
                                [
                                    IinPa + DiC * QinPae,
                                    DiC * IinPa + QinPae,
                                    ZiC * (CosC * UinPae + SinC * VinPa),
                                    -ZiC * (SinC * UinPae - CosC * VinPa),
                                ]
                            )
                            GT = np.dot(ATPe, IS1e)
                            GR = np.dot(ARPe, IS1e)
                            HT = np.dot(ATPe, IS2e)
                            HR = np.dot(ARPe, IS2e)
                    elif (
                        TypeC == 6
                    ):  # diattenuator calibration +-22.5° rotated_diattenuator_X22x5deg.odt
                        # parameters for calibration with aCal
                        AT = (
                            ATP1 * IinP
                            + sqr05 * DiC * (ATP1 * QinPe + ATP2e * IinP)
                            + (1 - 0.5 * WiC) * (ATP2e * QinPe + ATP3e * UinPe)
                            + ZiC
                            * (
                                sqr05 * SinC * (ATP3e * VinP - ATP4 * UinPe)
                                + ATP4 * CosC * VinP
                            )
                        )
                        BT = (
                            sqr05 * DiC * (ATP1 * UinPe + ATP3e * IinP)
                            + 0.5 * WiC * (ATP2e * UinPe + ATP3e * QinPe)
                            - sqr05 * ZiC * SinC * (ATP2e * VinP - ATP4 * QinPe)
                        )
                        AR = (
                            ARP1 * IinP
                            + sqr05 * DiC * (ARP1 * QinPe + ARP2e * IinP)
                            + (1 - 0.5 * WiC) * (ARP2e * QinPe + ARP3e * UinPe)
                            + ZiC
                            * (
                                sqr05 * SinC * (ARP3e * VinP - ARP4 * UinPe)
                                + ARP4 * CosC * VinP
                            )
                        )
                        BR = (
                            sqr05 * DiC * (ARP1 * UinPe + ARP3e * IinP)
                            + 0.5 * WiC * (ARP2e * UinPe + ARP3e * QinPe)
                            - sqr05 * ZiC * SinC * (ARP2e * VinP - ARP4 * QinPe)
                        )
                        # Correction parameters for normal measurements; they are independent of LDR
                        if (
                            not RotationErrorEpsilonForNormalMeasurements
                        ):  # calibrator taken out
                            IS1 = np.array([IinPo, QinPo, UinPo, VinPo])
                            IS2 = np.array([IinPa, QinPa, UinPa, VinPa])
                            GT = np.dot(ATP, IS1)
                            GR = np.dot(ARP, IS1)
                            HT = np.dot(ATP, IS2)
                            HR = np.dot(ARP, IS2)
                        else:
                            IS1e = np.array(
                                [
                                    IinPo + DiC * QinPoe,
                                    DiC * IinPo + QinPoe,
                                    ZiC * (CosC * UinPoe + SinC * VinPo),
                                    -ZiC * (SinC * UinPoe - CosC * VinPo),
                                ]
                            )
                            IS2e = np.array(
                                [
                                    IinPa + DiC * QinPae,
                                    DiC * IinPa + QinPae,
                                    ZiC * (CosC * UinPae + SinC * VinPa),
                                    -ZiC * (SinC * UinPae - CosC * VinPa),
                                ]
                            )
                            GT = np.dot(ATPe, IS1e)
                            GR = np.dot(ARPe, IS1e)
                            HT = np.dot(ATPe, IS2e)
                            HR = np.dot(ARPe, IS2e)
                    else:
                        print("Calibrator not implemented yet")
                        sys.exit()

                elif LocC == 3:  # C before receiver optics Eq.57

                    # S2ge = np.sin(np.deg2rad(2*RotO - 2*RotC))
                    # C2ge = np.cos(np.deg2rad(2*RotO - 2*RotC))
                    S2e = np.sin(np.deg2rad(2 * RotC))
                    C2e = np.cos(np.deg2rad(2 * RotC))

                    # AS with C before the receiver optics (see document rotated_diattenuator_X22x5deg.odt)
                    AF1 = np.array([1, C2g * DiO, S2g * DiO, 0])
                    AF2 = np.array(
                        [
                            C2g * DiO,
                            1 - S2g**2 * WiO,
                            S2g * C2g * WiO,
                            -S2g * ZiO * SinO,
                        ]
                    )
                    AF3 = np.array(
                        [
                            S2g * DiO,
                            S2g * C2g * WiO,
                            1 - C2g**2 * WiO,
                            C2g * ZiO * SinO,
                        ]
                    )
                    AF4 = np.array([0, S2g * SinO, -C2g * SinO, CosO])

                    ATF = ATP1 * AF1 + ATP2 * AF2 + ATP3 * AF3 + ATP4 * AF4
                    ARF = ARP1 * AF1 + ARP2 * AF2 + ARP3 * AF3 + ARP4 * AF4
                    ATF1 = ATF[0]
                    ATF2 = ATF[1]
                    ATF3 = ATF[2]
                    ATF4 = ATF[3]
                    ARF1 = ARF[0]
                    ARF2 = ARF[1]
                    ARF3 = ARF[2]
                    ARF4 = ARF[3]

                    # rotated AinF by epsilon
                    ATF2e = C2e * ATF[1] + S2e * ATF[2]
                    ATF3e = C2e * ATF[2] - S2e * ATF[1]
                    ARF2e = C2e * ARF[1] + S2e * ARF[2]
                    ARF3e = C2e * ARF[2] - S2e * ARF[1]

                    ATFe = np.array([ATF1, ATF2e, ATF3e, ATF4])
                    ARFe = np.array([ARF1, ARF2e, ARF3e, ARF4])

                    QinEe = C2e * QinE + S2e * UinE
                    UinEe = C2e * UinE - S2e * QinE

                    # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
                    IinF = IinE
                    QinF = aCal * QinE
                    UinF = -aCal * UinE
                    VinF = (1.0 - 2.0 * aCal) * VinE

                    IinFo = IinE
                    QinFo = 0.0
                    UinFo = 0.0
                    VinFo = VinE

                    IinFa = 0.0
                    QinFa = QinE
                    UinFa = -UinE
                    VinFa = -2.0 * VinE

                    # Stokes Input Vector before receiver optics rotated by epsilon Eq. C.3
                    QinFe = C2e * QinF + S2e * UinF
                    UinFe = C2e * UinF - S2e * QinF
                    QinFoe = C2e * QinFo + S2e * UinFo
                    UinFoe = C2e * UinFo - S2e * QinFo
                    QinFae = C2e * QinFa + S2e * UinFa
                    UinFae = C2e * UinFa - S2e * QinFa

                    # Calibration signals and Calibration correction K from measurements with LDRCal / aCal
                    if (TypeC == 2) or (TypeC == 1):  # rotator calibration Eq. C.4
                        AT = ATF1 * IinF + ATF4 * h * VinF
                        BT = ATF3e * QinF - ATF2e * h * UinF
                        AR = ARF1 * IinF + ARF4 * h * VinF
                        BR = ARF3e * QinF - ARF2e * h * UinF

                        # Correction parameters for normal measurements; they are independent of LDR
                        if not RotationErrorEpsilonForNormalMeasurements:
                            GT = ATF1 * IinE + ATF4 * VinE
                            GR = ARF1 * IinE + ARF4 * VinE
                            HT = ATF2 * QinE - ATF3 * UinE - ATF4 * 2 * VinE
                            HR = ARF2 * QinE - ARF3 * UinE - ARF4 * 2 * VinE
                        else:
                            GT = ATF1 * IinE + ATF4 * h * VinE
                            GR = ARF1 * IinE + ARF4 * h * VinE
                            HT = ATF2e * QinE - ATF3e * h * UinE - ATF4 * h * 2 * VinE
                            HR = ARF2e * QinE - ARF3e * h * UinE - ARF4 * h * 2 * VinE

                    elif (TypeC == 3) or (
                        TypeC == 4
                    ):  # linear polariser calibration Eq. C.5
                        # p = +45°, m = -45°
                        IF1e = np.array(
                            [IinF, ZiC * CosC * QinFe, UinFe, ZiC * CosC * VinF]
                        )
                        IF2e = np.array(
                            [
                                DiC * UinFe,
                                -ZiC * SinC * VinF,
                                DiC * IinF,
                                ZiC * SinC * QinFe,
                            ]
                        )

                        AT = np.dot(ATFe, IF1e)
                        AR = np.dot(ARFe, IF1e)
                        BT = np.dot(ATFe, IF2e)
                        BR = np.dot(ARFe, IF2e)

                        # Correction parameters for normal measurements; they are independent of LDR  --- the same as for TypeC = 6
                        if (
                            not RotationErrorEpsilonForNormalMeasurements
                        ):  # calibrator taken out
                            IS1 = np.array([IinE, 0, 0, VinE])
                            IS2 = np.array([0, QinE, -UinE, -2 * VinE])

                            GT = np.dot(ATF, IS1)
                            GR = np.dot(ARF, IS1)
                            HT = np.dot(ATF, IS2)
                            HR = np.dot(ARF, IS2)
                        else:
                            IS1e = np.array(
                                [
                                    IinFo + DiC * QinFoe,
                                    DiC * IinFo + QinFoe,
                                    ZiC * (CosC * UinFoe + SinC * VinFo),
                                    -ZiC * (SinC * UinFoe - CosC * VinFo),
                                ]
                            )
                            IS2e = np.array(
                                [
                                    IinFa + DiC * QinFae,
                                    DiC * IinFa + QinFae,
                                    ZiC * (CosC * UinFae + SinC * VinFa),
                                    -ZiC * (SinC * UinFae - CosC * VinFa),
                                ]
                            )
                            GT = np.dot(ATFe, IS1e)
                            GR = np.dot(ARFe, IS1e)
                            HT = np.dot(ATFe, IS2e)
                            HR = np.dot(ARFe, IS2e)

                    elif (
                        TypeC == 6
                    ):  # diattenuator calibration +-22.5° rotated_diattenuator_X22x5deg.odt
                        # p = +22.5°, m = -22.5°
                        IF1e = np.array(
                            [
                                IinF + sqr05 * DiC * QinFe,
                                sqr05 * DiC * IinF + (1 - 0.5 * WiC) * QinFe,
                                (1 - 0.5 * WiC) * UinFe + sqr05 * ZiC * SinC * VinF,
                                -sqr05 * ZiC * SinC * UinFe + ZiC * CosC * VinF,
                            ]
                        )
                        IF2e = np.array(
                            [
                                sqr05 * DiC * UinFe,
                                0.5 * WiC * UinFe - sqr05 * ZiC * SinC * VinF,
                                sqr05 * DiC * IinF + 0.5 * WiC * QinFe,
                                sqr05 * ZiC * SinC * QinFe,
                            ]
                        )

                        AT = np.dot(ATFe, IF1e)
                        AR = np.dot(ARFe, IF1e)
                        BT = np.dot(ATFe, IF2e)
                        BR = np.dot(ARFe, IF2e)

                        # Correction parameters for normal measurements; they are independent of LDR
                        if (
                            not RotationErrorEpsilonForNormalMeasurements
                        ):  # calibrator taken out
                            # IS1 = np.array([IinE,0,0,VinE])
                            # IS2 = np.array([0,QinE,-UinE,-2*VinE])
                            IS1 = np.array([IinFo, 0, 0, VinFo])
                            IS2 = np.array([0, QinFa, UinFa, VinFa])
                            GT = np.dot(ATF, IS1)
                            GR = np.dot(ARF, IS1)
                            HT = np.dot(ATF, IS2)
                            HR = np.dot(ARF, IS2)
                        else:
                            # IS1e = np.array([IinE,DiC*IinE,ZiC*SinC*VinE,ZiC*CosC*VinE])
                            # IS2e = np.array([DiC*QinEe,QinEe,-ZiC*(CosC*UinEe+2*SinC*VinE),ZiC*(SinC*UinEe-2*CosC*VinE)])
                            IS1e = np.array(
                                [
                                    IinFo + DiC * QinFoe,
                                    DiC * IinFo + QinFoe,
                                    ZiC * (CosC * UinFoe + SinC * VinFo),
                                    -ZiC * (SinC * UinFoe - CosC * VinFo),
                                ]
                            )
                            IS2e = np.array(
                                [
                                    IinFa + DiC * QinFae,
                                    DiC * IinFa + QinFae,
                                    ZiC * (CosC * UinFae + SinC * VinFa),
                                    -ZiC * (SinC * UinFae - CosC * VinFa),
                                ]
                            )
                            GT = np.dot(ATFe, IS1e)
                            GR = np.dot(ARFe, IS1e)
                            HT = np.dot(ATFe, IS2e)
                            HR = np.dot(ARFe, IS2e)

                    else:
                        print("Calibrator not implemented yet")
                        sys.exit()

                elif LocC == 2:  # C behind emitter optics Eq.57
                    # print("Calibrator location not implemented yet")
                    S2e = np.sin(np.deg2rad(2 * RotC))
                    C2e = np.cos(np.deg2rad(2 * RotC))

                    # AS with C before the receiver optics (see document rotated_diattenuator_X22x5deg.odt)
                    AF1 = np.array([1, C2g * DiO, S2g * DiO, 0])
                    AF2 = np.array(
                        [
                            C2g * DiO,
                            1 - S2g**2 * WiO,
                            S2g * C2g * WiO,
                            -S2g * ZiO * SinO,
                        ]
                    )
                    AF3 = np.array(
                        [
                            S2g * DiO,
                            S2g * C2g * WiO,
                            1 - C2g**2 * WiO,
                            C2g * ZiO * SinO,
                        ]
                    )
                    AF4 = np.array([0, S2g * SinO, -C2g * SinO, CosO])

                    ATF = ATP1 * AF1 + ATP2 * AF2 + ATP3 * AF3 + ATP4 * AF4
                    ARF = ARP1 * AF1 + ARP2 * AF2 + ARP3 * AF3 + ARP4 * AF4
                    ATF1 = ATF[0]
                    ATF2 = ATF[1]
                    ATF3 = ATF[2]
                    ATF4 = ATF[3]
                    ARF1 = ARF[0]
                    ARF2 = ARF[1]
                    ARF3 = ARF[2]
                    ARF4 = ARF[3]

                    # AS with C behind the emitter  --------------------------------------------
                    # terms without aCal
                    ATE1o, ARE1o = ATF1, ARF1
                    ATE2o, ARE2o = 0.0, 0.0
                    ATE3o, ARE3o = 0.0, 0.0
                    ATE4o, ARE4o = ATF4, ARF4
                    # terms with aCal
                    ATE1a, ARE1a = 0.0, 0.0
                    ATE2a, ARE2a = ATF2, ARF2
                    ATE3a, ARE3a = -ATF3, -ARF3
                    ATE4a, ARE4a = -2 * ATF4, -2 * ARF4
                    # rotated AinEa by epsilon
                    ATE2ae = C2e * ATF2 + S2e * ATF3
                    ATE3ae = -S2e * ATF2 - C2e * ATF3
                    ARE2ae = C2e * ARF2 + S2e * ARF3
                    ARE3ae = -S2e * ARF2 - C2e * ARF3

                    ATE1 = ATE1o
                    ATE2e = aCal * ATE2ae
                    ATE3e = aCal * ATE3ae
                    ATE4 = (1 - 2 * aCal) * ATF4
                    ARE1 = ARE1o
                    ARE2e = aCal * ARE2ae
                    ARE3e = aCal * ARE3ae
                    ARE4 = (1.0 - 2.0 * aCal) * ARF4

                    # rotated IinE
                    QinEe = C2e * QinE + S2e * UinE
                    UinEe = C2e * UinE - S2e * QinE

                    # --- Calibration signals and Calibration correction K from measurements with LDRCal / aCal
                    if (TypeC == 2) or (
                        TypeC == 1
                    ):  # +++++++++ rotator calibration Eq. C.4
                        AT = ATE1o * IinE + (ATE4o + aCal * ATE4a) * h * VinE
                        BT = aCal * (ATE3ae * QinEe - ATE2ae * h * UinEe)
                        AR = ARE1o * IinE + (ARE4o + aCal * ARE4a) * h * VinE
                        BR = aCal * (ARE3ae * QinEe - ARE2ae * h * UinEe)

                        # Correction parameters for normal measurements; they are independent of LDR
                        if not RotationErrorEpsilonForNormalMeasurements:
                            # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
                            GT = ATE1o * IinE + ATE4o * h * VinE
                            GR = ARE1o * IinE + ARE4o * h * VinE
                            HT = ATE2a * QinE + ATE3a * h * UinEe + ATE4a * h * VinE
                            HR = ARE2a * QinE + ARE3a * h * UinEe + ARE4a * h * VinE
                        else:
                            GT = ATE1o * IinE + ATE4o * h * VinE
                            GR = ARE1o * IinE + ARE4o * h * VinE
                            HT = ATE2ae * QinE + ATE3ae * h * UinEe + ATE4a * h * VinE
                            HR = ARE2ae * QinE + ARE3ae * h * UinEe + ARE4a * h * VinE

                    elif (TypeC == 3) or (
                        TypeC == 4
                    ):  # +++++++++ linear polariser calibration Eq. C.5
                        # p = +45°, m = -45°
                        AT = (
                            ATE1 * IinE
                            + ZiC * CosC * (ATE2e * QinEe + ATE4 * VinE)
                            + ATE3e * UinEe
                        )
                        BT = DiC * (ATE1 * UinEe + ATE3e * IinE) + ZiC * SinC * (
                            ATE4 * QinEe - ATE2e * VinE
                        )
                        AR = (
                            ARE1 * IinE
                            + ZiC * CosC * (ARE2e * QinEe + ARE4 * VinE)
                            + ARE3e * UinEe
                        )
                        BR = DiC * (ARE1 * UinEe + ARE3e * IinE) + ZiC * SinC * (
                            ARE4 * QinEe - ARE2e * VinE
                        )

                        # Correction parameters for normal measurements; they are independent of LDR
                        if not RotationErrorEpsilonForNormalMeasurements:
                            # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
                            GT = ATE1o * IinE + ATE4o * VinE
                            GR = ARE1o * IinE + ARE4o * VinE
                            HT = ATE2a * QinE + ATE3a * UinE + ATE4a * VinE
                            HR = ARE2a * QinE + ARE3a * UinE + ARE4a * VinE
                        else:
                            D = IinE + DiC * QinEe
                            A = DiC * IinE + QinEe
                            B = ZiC * (CosC * UinEe + SinC * VinE)
                            C = -ZiC * (SinC * UinEe - CosC * VinE)
                            GT = ATE1o * D + ATE4o * C
                            GR = ARE1o * D + ARE4o * C
                            HT = ATE2a * A + ATE3a * B + ATE4a * C
                            HR = ARE2a * A + ARE3a * B + ARE4a * C

                    elif (
                        TypeC == 6
                    ):  # real HWP calibration +-22.5° rotated_diattenuator_X22x5deg.odt
                        # p = +22.5°, m = -22.5°
                        IE1e = np.array(
                            [
                                IinE + sqr05 * DiC * QinEe,
                                sqr05 * DiC * IinE + (1 - 0.5 * WiC) * QinEe,
                                (1.0 - 0.5 * WiC) * UinEe + sqr05 * ZiC * SinC * VinE,
                                -sqr05 * ZiC * SinC * UinEe + ZiC * CosC * VinE,
                            ]
                        )
                        IE2e = np.array(
                            [
                                sqr05 * DiC * UinEe,
                                0.5 * WiC * UinEe - sqr05 * ZiC * SinC * VinE,
                                sqr05 * DiC * IinE + 0.5 * WiC * QinEe,
                                sqr05 * ZiC * SinC * QinEe,
                            ]
                        )
                        ATEe = np.array([ATE1, ATE2e, ATE3e, ATE4])
                        AREe = np.array([ARE1, ARE2e, ARE3e, ARE4])
                        AT = np.dot(ATEe, IE1e)
                        AR = np.dot(AREe, IE1e)
                        BT = np.dot(ATEe, IE2e)
                        BR = np.dot(AREe, IE2e)

                        # Correction parameters for normal measurements; they are independent of LDR
                        if (
                            not RotationErrorEpsilonForNormalMeasurements
                        ):  # calibrator taken out
                            GT = ATE1o * IinE + ATE4o * VinE
                            GR = ARE1o * IinE + ARE4o * VinE
                            HT = ATE2a * QinE + ATE3a * UinE + ATE4a * VinE
                            HR = ARE2a * QinE + ARE3a * UinE + ARE4a * VinE
                        else:
                            D = IinE + DiC * QinEe
                            A = DiC * IinE + QinEe
                            B = ZiC * (CosC * UinEe + SinC * VinE)
                            C = -ZiC * (SinC * UinEe - CosC * VinE)
                            GT = ATE1o * D + ATE4o * C
                            GR = ARE1o * D + ARE4o * C
                            HT = ATE2a * A + ATE3a * B + ATE4a * C
                            HR = ARE2a * A + ARE3a * B + ARE4a * C
                    else:
                        print("Calibrator not implemented yet")
                        sys.exit()

                for iTCalT, iTCalR, iNCalTp, iNCalTm, iNCalRp, iNCalRm, iNIt, iNIr in [
                    (iTCalT, iTCalR, iNCalTp, iNCalTm, iNCalRp, iNCalRm, iNIt, iNIr)
                    for iTCalT in range(-nTCalT, nTCalT + 1)  # Etax
                    for iTCalR in range(-nTCalR, nTCalR + 1)  # Etax
                    for iNCalTp in range(
                        -nNCal, nNCal + 1
                    )  # noise error of calibration signals => Etax
                    for iNCalTm in range(
                        -nNCal, nNCal + 1
                    )  # noise error of calibration signals => Etax
                    for iNCalRp in range(
                        -nNCal, nNCal + 1
                    )  # noise error of calibration signals => Etax
                    for iNCalRm in range(
                        -nNCal, nNCal + 1
                    )  # noise error of calibration signals => Etax
                    for iNIt in range(-nNI, nNI + 1)
                    for iNIr in range(-nNI, nNI + 1)
                ]:

                    # Calibration signals with aCal => Determination of the correction K of the real calibration factor
                    IoutTp = TTa * TiC * TiO * TiE * (AT + BT)
                    IoutTm = TTa * TiC * TiO * TiE * (AT - BT)
                    IoutRp = TRa * TiC * TiO * TiE * (AR + BR)
                    IoutRm = TRa * TiC * TiO * TiE * (AR - BR)

                    if nTCalT > 0:
                        TCalT = TCalT0 + iTCalT * dTCalT / nTCalT
                    if nTCalR > 0:
                        TCalR = TCalR0 + iTCalR * dTCalR / nTCalR
                    # signal noise errors
                    # ----- random error calculation ----------
                    # noise must be calculated from/with the actually measured signals; influence of TCalT, TCalR errors on noise are not considered ?
                    # actually measured signal counts are in input file and don't change
                    # relative standard deviation of calibration signals with LDRcal; assumed to be statisitcally independent
                    # error nNCal: one-sigma in steps to left and right for calibration signals
                    if nNCal > 0:
                        if CalcFrom0deg:
                            dIoutTp = (NCalT * IoutTp) ** -0.5
                            dIoutTm = (NCalT * IoutTm) ** -0.5
                            dIoutRp = (NCalR * IoutRp) ** -0.5
                            dIoutRm = (NCalR * IoutRm) ** -0.5
                        else:
                            dIoutTp = dIoutTp0 * (IoutTp / IoutTp0)
                            dIoutTm = dIoutTm0 * (IoutTm / IoutTm0)
                            dIoutRp = dIoutRp0 * (IoutRp / IoutRp0)
                            dIoutRm = dIoutRm0 * (IoutRm / IoutRm0)
                        # print(iTCalT, iTCalR, iNCalTp, iNCalTm, iNCalRp, iNCalRm, iNIt, iNIr, IoutTp, dIoutTp)
                        IoutTp = IoutTp * (1.0 + iNCalTp * dIoutTp / nNCal)
                        IoutTm = IoutTm * (1.0 + iNCalTm * dIoutTm / nNCal)
                        IoutRp = IoutRp * (1.0 + iNCalRp * dIoutRp / nNCal)
                        IoutRm = IoutRm * (1.0 + iNCalRm * dIoutRm / nNCal)

                    IoutTp = IoutTp * TCalT / TCalT0
                    IoutTm = IoutTm * TCalT / TCalT0
                    IoutRp = IoutRp * TCalR / TCalR0
                    IoutRm = IoutRm * TCalR / TCalR0
                    # --- Results and Corrections; electronic etaR and etaT are assumed to be 1 for true and assumed true systems
                    # calibration factor
                    Eta = (
                        TRa / TTa
                    )  # = TRa / TTa; Eta = Eta*/K  Eq. 84; corrected according to the papers supplement Eqs. (S.10.10.1) ff
                    # possibly real calibration factor
                    Etapx = IoutRp / IoutTp
                    Etamx = IoutRm / IoutTm
                    Etax = (Etapx * Etamx) ** 0.5
                    K = Etax / Eta
                    # print("{0:6.3f},{1:6.3f},{2:6.3f},{3:6.3f},{4:6.3f},{5:6.3f},{6:6.3f},{7:6.3f},{8:6.3f},{9:6.3f},{10:6.3f}".format(AT, BT, AR, BR, DiC, ZiC, RetO, TP, TS, Kp, Km))
                    # print("{0:6.3f},{1:6.3f},{2:6.3f},{3:6.3f}".format(DiC, ZiC, Kp, Km))

                    #  For comparison with Volkers Libreoffice Müller Matrix spreadsheet
                    # Eta_test_p = (IoutRp/IoutTp)
                    # Eta_test_m = (IoutRm/IoutTm)
                    # Eta_test = (Eta_test_p*Eta_test_m)**0.5
                    """
                    for iIt, iIr \
                            in [(iIt, iIr)
                                for iIt in range(-nNI, nNI + 1)
                                for iIr in range(-nNI, nNI + 1)]:
                    """

                    iN = iN + 1
                    if iN == 10001:
                        ctime = clock()
                        print(
                            " estimated time ",
                            "{0:4.2f}".format(N / 10000 * (ctime - atime)),
                            "sec ",
                        )  # , end="")
                        print(
                            "\r elapsed time ",
                            "{0:5.0f}".format((ctime - atime)),
                            "sec ",
                            end="\r",
                        )
                    ctime = clock()
                    if (ctime - dtime) > 10:
                        print(
                            "\r elapsed time ",
                            "{0:5.0f}".format((ctime - atime)),
                            "sec ",
                            end="\r",
                        )
                        dtime = ctime

                    # *** loop for different real LDRs **********************************************************************
                    iLDR = -1
                    for LDRTrue in LDRrange:
                        iLDR = iLDR + 1
                        atrue = (1.0 - LDRTrue) / (1.0 + LDRTrue)
                        # ----- Forward simulated signals and LDRsim with atrue; from input file; not considering TiC.
                        It = (
                            TTa * TiO * TiE * (GT + atrue * HT)
                        )  # TaT*TiT*TiC*TiO*IinL*(GT+atrue*HT)
                        Ir = (
                            TRa * TiO * TiE * (GR + atrue * HR)
                        )  # TaR*TiR*TiC*TiO*IinL*(GR+atrue*HR)
                        # # signal noise errors; standard deviation of signals; assumed to be statisitcally independent
                        # because the signals depend on LDRtrue, the errors dIt and dIr must be calculated for each LDRtrue
                        if CalcFrom0deg:
                            """
                            dIt = ((NCalT * It / IoutTp * NILfac / TCalT) ** -0.5)
                            dIr = ((NCalR * Ir / IoutRp * NILfac / TCalR) ** -0.5)
                            """
                            dIt = (It * NI * eFacT) ** -0.5
                            dIr = (Ir * NI * eFacR) ** -0.5
                        else:
                            dIt = (It * NI * eFacT) ** -0.5
                            dIr = (Ir * NI * eFacR) ** -0.5
                            """
                            # does this work? Why not as above?
                            dIt = ((NCalT * 2. * NILfac / TCalT ) ** -0.5)
                            dIr = ((NCalR * 2. * NILfac / TCalR) ** -0.5)
                            """
                        # error nNI: one-sigma in steps to left and right for 0° signals
                        if nNI > 0:
                            It = It * (1.0 + iNIt * dIt / nNI)
                            Ir = Ir * (1.0 + iNIr * dIr / nNI)

                        # LDRsim = 1/Eta*Ir/It  # simulated LDR* with Y from input file
                        LDRsim = (
                            Ir / It
                        )  # simulated uncorrected LDR with Y from input file

                        # ----- Backward correction
                        # Corrected LDRCorr  with assumed true G0,H0,K0,Eta0 from forward simulated (real) LDRsim(atrue)
                        LDRCorr = (LDRsim / (Etax / K0) * (GT0 + HT0) - (GR0 + HR0)) / (
                            (GR0 - HR0) - LDRsim / (Etax / K0) * (GT0 - HT0)
                        )

                        # The following is a test whether the equations for calibration Etax and normal  signal (GHK, LDRsim) are consistent
                        # LDRCorr = (LDRsim / Eta * (GT + HT) - (GR + HR)) / ((GR - HR) - LDRsim / Eta * (GT - HT))
                        # Without any correction
                        LDRunCorr = LDRsim / Etax
                        # LDRunCorr = (LDRsim / Etax * (GT / abs(GT) + HT / abs(HT)) - (GR / abs(GR) + HR / abs(HR))) / ((GR / abs(GR) - HR / abs(HR)) - LDRsim / Etax * (GT / abs(GT) - HT / abs(HT)))

                        """
                        # -- F11corr from It and Ir and calibration EtaX
                        Text1 = "!!! EXPERIMENTAL !!!  F11corr from It and Ir with calibration EtaX: x-axis: F11corr(LDRtrue) / F11corr(LDRtrue = 0.004) - 1"
                        F11corr = 1 / (TiO * TiE) * (
                        (HR0 * Etax / K0 * It / TTa - HT0 * Ir / TRa) / (HR0 * GT0 - HT0 * GR0))  # IL = 1  Eq.(64); Etax/K0 = Eta0.
                        """
                        # Corrected F11corr  with assumed true G0,H0,K0 from forward simulated (real) It and Ir (atrue)
                        Text1 = "!!! EXPERIMENTAL !!!  F11corr from real It and Ir with real calibration EtaX: x-axis: F11corr(LDRtrue) / aF11sim0(LDRtrue) - 1"
                        F11corr = (
                            1
                            / (TiO * TiE)
                            * (
                                (HR0 * Etax / K0 * It / TTa - HT0 * Ir / TRa)
                                / (HR0 * GT0 - HT0 * GR0)
                            )
                        )  # IL = 1  Eq.(64); Etax/K0 = Eta0.

                        # Text1 = "F11corr from It and Ir without corrections but with calibration EtaX: x-axis: F11corr(LDRtrue) devided by F11corr(LDRtrue = 0.004)"
                        # F11corr = 0.5/(TiO*TiE)*(Etax*It/TTa+Ir/TRa)    # IL = 1  Eq.(64)

                        # -- It from It only with atrue without corrections - for BERTHA (and PollyXTs)
                        # Text1 = " x-axis: IT(LDRtrue) / IT(LDRtrue = 0.004) - 1"
                        # F11corr = It/(TaT*TiT*TiO*TiE)   #/(TaT*TiT*TiO*TiE*(GT0+atrue*HT0))
                        # ! see below line 1673ff

                        aF11corr[iLDR, iN] = F11corr
                        aLDRcorr[iLDR, iN] = LDRCorr  # LDRCorr # LDRsim # for test only
                        aLDRsim[iLDR, iN] = LDRsim  # LDRCorr # LDRsim # for test only
                        # aPLDR[iLDR, iN] = CalcPLDR(LDRCorr, BSR[iLDR], LDRm0)
                        aEtax[iLDR, iN] = Etax
                        aEtapx[iLDR, iN] = Etapx
                        aEtamx[iLDR, iN] = Etamx

                        aGHK[0, iN] = GR
                        aGHK[1, iN] = GT
                        aGHK[2, iN] = HR
                        aGHK[3, iN] = HT
                        aGHK[4, iN] = K

                        aLDRCal[iN] = iLDRCal
                        aQin[iN] = iQin
                        aVin[iN] = iVin
                        aERaT[iN] = iERaT
                        aERaR[iN] = iERaR
                        aRotaT[iN] = iRotaT
                        aRotaR[iN] = iRotaR
                        aRetT[iN] = iRetT
                        aRetR[iN] = iRetR

                        aRotL[iN] = iRotL
                        aRotE[iN] = iRotE
                        aRetE[iN] = iRetE
                        aRotO[iN] = iRotO
                        aRetO[iN] = iRetO
                        aRotC[iN] = iRotC
                        aRetC[iN] = iRetC
                        aDiO[iN] = iDiO
                        aDiE[iN] = iDiE
                        aDiC[iN] = iDiC
                        aTP[iN] = iTP
                        aTS[iN] = iTS
                        aRP[iN] = iRP
                        aRS[iN] = iRS
                        aTCalT[iN] = iTCalT
                        aTCalR[iN] = iTCalR

                        aNCalTp[iN] = iNCalTp  # IoutTp, IoutTm, IoutRp, IoutRm => Etax
                        aNCalTm[iN] = iNCalTm  # IoutTp, IoutTm, IoutRp, IoutRm => Etax
                        aNCalRp[iN] = iNCalRp  # IoutTp, IoutTm, IoutRp, IoutRm => Etax
                        aNCalRm[iN] = iNCalRm  # IoutTp, IoutTm, IoutRp, IoutRm => Etax
                        aNIt[iN] = iNIt  # It, Tr
                        aNIr[iN] = iNIr  # It, Tr

    # --- END loop
    btime = clock()
    # print("\r done in      ", "{0:5.0f}".format(btime - atime), "sec.      => producing plots now .... some more seconds ..."),  # , end="\r");
    print(
        " done in      ",
        "{0:5.0f}".format(btime - atime),
        "sec.      => producing plots now .... some more seconds ...",
    )
    # --- Plot -----------------------------------------------------------------
    # print("Errors from GHK correction uncertainties:")
    if sns_loaded:
        sns.set_style("whitegrid")
        sns.set_palette("bright6", 6)
        # for older seaborn versions use:
        # sns.set_palette("bright", 6)

    """
    fig2 = plt.figure()
    plt.plot(aLDRcorr[2,:],'b.')
    plt.plot(aLDRcorr[3,:],'r.')
    plt.plot(aLDRcorr[4,:],'g.')
    #plt.plot(aLDRcorr[6,:],'c.')
    plt.show
    """

    # Plot LDR
    def PlotSubHist(aVar, aX, X0, daX, iaX, naX):
        # aVar is the name of the parameter and aX is the subset of aLDRcorr which is coloured in the plot
        # example: PlotSubHist("DOLP", aDOLP, DOLP0, dDOLP, iDOLP, nDOLP)
        fig, ax = plt.subplots(
            nrows=1, ncols=5, sharex=True, sharey=True, figsize=(25, 2)
        )
        iLDR = -1
        for LDRTrue in LDRrange:
            aXmean = np.zeros(2 * naX + 1)
            iLDR = iLDR + 1
            LDRmin[iLDR] = np.amin(aLDRcorr[iLDR, :])
            LDRmax[iLDR] = np.amax(aLDRcorr[iLDR, :])
            if LDRmax[iLDR] > 10:
                LDRmax[iLDR] = 10
            if LDRmin[iLDR] < -10:
                LDRmin[iLDR] = -10
            Rmin = LDRmin[iLDR] * 0.995  # np.min(aLDRcorr[iLDR,:])    * 0.995
            Rmax = LDRmax[iLDR] * 1.005  # np.max(aLDRcorr[iLDR,:])    * 1.005

            # Determine mean distance of all aXmean from each other for each iLDR
            meanDist = 0.0
            for iaX in range(-naX, naX + 1):
                # mean LDRCorr value for certain error (iaX) of parameter aVar
                aXmean[iaX + naX] = np.mean(aLDRcorr[iLDR, aX == iaX])
            # relative to absolute spread of LDRCorrs
            meanDist = (
                (np.max(aXmean) - np.min(aXmean)) / (LDRmax[iLDR] - LDRmin[iLDR]) * 100
            )

            plt.subplot(1, 5, iLDR + 1)
            (n, bins, patches) = plt.hist(
                aLDRcorr[iLDR, :],
                bins=100,
                log=False,
                range=[Rmin, Rmax],
                alpha=0.5,
                density=False,
                color="0.5",
                histtype="stepfilled",
            )

            for iaX in range(-naX, naX + 1):
                # mean LDRCorr value for certain error (iaX) of parameter aVar
                plt.hist(
                    aLDRcorr[iLDR, aX == iaX],
                    range=[Rmin, Rmax],
                    bins=100,
                    log=False,
                    alpha=0.3,
                    density=False,
                    histtype="stepfilled",
                    label=str(round(X0 + iaX * daX / naX, 5)),
                )

                if iLDR == 2:
                    leg = plt.legend()
                    leg.get_frame().set_alpha(0.1)

            plt.tick_params(axis="both", labelsize=10)
            plt.plot([LDRTrue, LDRTrue], [0, np.max(n)], "r-", lw=2)
            plt.gca().set_title("{0:3.0f}%".format(meanDist))
            plt.gca().set_xlabel("LDRtrue", color="red")

        # plt.ylabel('frequency', fontsize=10)
        # plt.xlabel('LDRCorr', fontsize=10)
        # fig.tight_layout()
        fig.suptitle(
            LID
            + " with "
            + str(Type[TypeC])
            + " "
            + str(Loc[LocC])
            + " - "
            + aVar
            + " error contribution",
            fontsize=14,
            y=1.10,
        )
        # plt.show()
        # fig.savefig(LID + '_' + aVar + '.png', dpi=150, bbox_inches='tight', pad_inches=0)
        # plt.close
        return

    def PlotLDRsim(aVar, aX, X0, daX, iaX, naX):
        # aVar is the name of the parameter and aX is the subset of aLDRsim which is coloured in the plot
        # example: PlotSubHist("DOLP", aDOLP, DOLP0, dDOLP, iDOLP, nDOLP)
        fig, ax = plt.subplots(
            nrows=1, ncols=5, sharex=True, sharey=True, figsize=(25, 2)
        )
        iLDR = -1
        for LDRTrue in LDRrange:
            aXmean = np.zeros(2 * naX + 1)
            iLDR = iLDR + 1
            LDRsimmin[iLDR] = np.amin(aLDRsim[iLDR, :])
            LDRsimmax[iLDR] = np.amax(aLDRsim[iLDR, :])
            # print("LDRsimmin[iLDR], LDRsimmax[iLDR] = ", LDRsimmin[iLDR], LDRsimmax[iLDR])
            # if (LDRsimmax[iLDR] > 10): LDRsimmax[iLDR] = 10
            # if (LDRsimmin[iLDR] < -10): LDRsimmin[iLDR] = -10
            Rmin = LDRsimmin[iLDR] * 0.995  # np.min(aLDRsim[iLDR,:])    * 0.995
            Rmax = LDRsimmax[iLDR] * 1.005  # np.max(aLDRsim[iLDR,:])    * 1.005
            # print("Rmin, Rmax = ", Rmin, Rmax)

            # Determine mean distance of all aXmean from each other for each iLDR
            meanDist = 0.0
            for iaX in range(-naX, naX + 1):
                # mean LDRCorr value for certain error (iaX) of parameter aVar
                aXmean[iaX + naX] = np.mean(aLDRsim[iLDR, aX == iaX])
            # relative to absolute spread of LDRCorrs
            meanDist = (
                (np.max(aXmean) - np.min(aXmean))
                / (LDRsimmax[iLDR] - LDRsimmin[iLDR])
                * 100
            )

            plt.subplot(1, 5, iLDR + 1)
            (n, bins, patches) = plt.hist(
                aLDRsim[iLDR, :],
                bins=100,
                log=False,
                range=[Rmin, Rmax],
                alpha=0.5,
                density=False,
                color="0.5",
                histtype="stepfilled",
            )

            for iaX in range(-naX, naX + 1):
                # mean LDRCorr value for certain error (iaX) of parameter aVar
                plt.hist(
                    aLDRsim[iLDR, aX == iaX],
                    range=[Rmin, Rmax],
                    bins=100,
                    log=False,
                    alpha=0.3,
                    density=False,
                    histtype="stepfilled",
                    label=str(round(X0 + iaX * daX / naX, 5)),
                )

                if iLDR == 2:
                    leg = plt.legend()
                    leg.get_frame().set_alpha(0.1)

            plt.tick_params(axis="both", labelsize=10)
            plt.plot([LDRsim0[iLDR], LDRsim0[iLDR]], [0, np.max(n)], "r-", lw=2)
            plt.gca().set_title("{0:3.0f}%".format(meanDist))
            plt.gca().set_xlabel("LDRsim0", color="red")

        fig.suptitle(
            "LDRsim - "
            + LID
            + " with "
            + str(Type[TypeC])
            + " "
            + str(Loc[LocC])
            + " - "
            + aVar
            + " error contribution",
            fontsize=14,
            y=1.10,
        )
        return

    # Plot Etax
    def PlotEtax(aVar, aX, X0, daX, iaX, naX):
        # aVar is the name of the parameter and aX is the subset of aLDRcorr which is coloured in the plot
        # example: PlotSubHist("DOLP", aDOLP, DOLP0, dDOLP, iDOLP, nDOLP)
        fig, ax = plt.subplots(
            nrows=1, ncols=5, sharex=True, sharey=True, figsize=(25, 2)
        )
        iLDR = -1
        for LDRTrue in LDRrange:
            aXmean = np.zeros(2 * naX + 1)
            iLDR = iLDR + 1
            Etaxmin = np.amin(aEtax[iLDR, :])
            Etaxmax = np.amax(aEtax[iLDR, :])
            Rmin = Etaxmin * 0.995  # np.min(aLDRcorr[iLDR,:])    * 0.995
            Rmax = Etaxmax * 1.005  # np.max(aLDRcorr[iLDR,:])    * 1.005

            # Determine mean distance of all aXmean from each other for each iLDR
            meanDist = 0.0
            for iaX in range(-naX, naX + 1):
                # mean Etax value for certain error (iaX) of parameter aVar
                aXmean[iaX + naX] = np.mean(aEtax[iLDR, aX == iaX])
            # relative to absolute spread of Etax
            meanDist = (np.max(aXmean) - np.min(aXmean)) / (Etaxmax - Etaxmin) * 100

            plt.subplot(1, 5, iLDR + 1)
            (n, bins, patches) = plt.hist(
                aEtax[iLDR, :],
                bins=50,
                log=False,
                range=[Rmin, Rmax],
                alpha=0.5,
                density=False,
                color="0.5",
                histtype="stepfilled",
            )
            for iaX in range(-naX, naX + 1):
                plt.hist(
                    aEtax[iLDR, aX == iaX],
                    range=[Rmin, Rmax],
                    bins=50,
                    log=False,
                    alpha=0.3,
                    density=False,
                    histtype="stepfilled",
                    label=str(round(X0 + iaX * daX / naX, 5)),
                )
                if iLDR == 2:
                    leg = plt.legend()
                    leg.get_frame().set_alpha(0.1)
            plt.tick_params(axis="both", labelsize=10)
            plt.plot([Etax0, Etax0], [0, np.max(n)], "r-", lw=2)
            plt.gca().set_title("{0:3.0f}%".format(meanDist))
            plt.gca().set_xlabel("Etax0", color="red")
        fig.suptitle(
            "Etax - "
            + LID
            + " with "
            + str(Type[TypeC])
            + " "
            + str(Loc[LocC])
            + " - "
            + aVar
            + " error contribution",
            fontsize=14,
            y=1.10,
        )
        return

    def PlotEtapx(aVar, aX, X0, daX, iaX, naX):
        # aVar is the name of the parameter and aX is the subset of aLDRcorr which is coloured in the plot
        # example: PlotSubHist("DOLP", aDOLP, DOLP0, dDOLP, iDOLP, nDOLP)
        fig, ax = plt.subplots(
            nrows=1, ncols=5, sharex=True, sharey=True, figsize=(25, 2)
        )
        iLDR = -1
        for LDRTrue in LDRrange:
            aXmean = np.zeros(2 * naX + 1)
            iLDR = iLDR + 1
            Etapxmin = np.amin(aEtapx[iLDR, :])
            Etapxmax = np.amax(aEtapx[iLDR, :])
            Rmin = Etapxmin * 0.995  # np.min(aLDRcorr[iLDR,:])    * 0.995
            Rmax = Etapxmax * 1.005  # np.max(aLDRcorr[iLDR,:])    * 1.005

            # Determine mean distance of all aXmean from each other for each iLDR
            meanDist = 0.0
            for iaX in range(-naX, naX + 1):
                # mean Etapx value for certain error (iaX) of parameter aVar
                aXmean[iaX + naX] = np.mean(aEtapx[iLDR, aX == iaX])
            # relative to absolute spread of Etapx
            meanDist = (np.max(aXmean) - np.min(aXmean)) / (Etapxmax - Etapxmin) * 100

            plt.subplot(1, 5, iLDR + 1)
            (n, bins, patches) = plt.hist(
                aEtapx[iLDR, :],
                bins=50,
                log=False,
                range=[Rmin, Rmax],
                alpha=0.5,
                density=False,
                color="0.5",
                histtype="stepfilled",
            )
            for iaX in range(-naX, naX + 1):
                plt.hist(
                    aEtapx[iLDR, aX == iaX],
                    range=[Rmin, Rmax],
                    bins=50,
                    log=False,
                    alpha=0.3,
                    density=False,
                    histtype="stepfilled",
                    label=str(round(X0 + iaX * daX / naX, 5)),
                )
                if iLDR == 2:
                    leg = plt.legend()
                    leg.get_frame().set_alpha(0.1)
            plt.tick_params(axis="both", labelsize=10)
            plt.plot([Etapx0, Etapx0], [0, np.max(n)], "r-", lw=2)
            plt.gca().set_title("{0:3.0f}%".format(meanDist))
            plt.gca().set_xlabel("Etapx0", color="red")
        fig.suptitle(
            "Etapx - "
            + LID
            + " with "
            + str(Type[TypeC])
            + " "
            + str(Loc[LocC])
            + " - "
            + aVar
            + " error contribution",
            fontsize=14,
            y=1.10,
        )
        return

    def PlotEtamx(aVar, aX, X0, daX, iaX, naX):
        # aVar is the name of the parameter and aX is the subset of aLDRcorr which is coloured in the plot
        # example: PlotSubHist("DOLP", aDOLP, DOLP0, dDOLP, iDOLP, nDOLP)
        fig, ax = plt.subplots(
            nrows=1, ncols=5, sharex=True, sharey=True, figsize=(25, 2)
        )
        iLDR = -1
        for LDRTrue in LDRrange:
            aXmean = np.zeros(2 * naX + 1)
            iLDR = iLDR + 1
            Etamxmin = np.amin(aEtamx[iLDR, :])
            Etamxmax = np.amax(aEtamx[iLDR, :])
            Rmin = Etamxmin * 0.995  # np.min(aLDRcorr[iLDR,:])    * 0.995
            Rmax = Etamxmax * 1.005  # np.max(aLDRcorr[iLDR,:])    * 1.005

            # Determine mean distance of all aXmean from each other for each iLDR
            meanDist = 0.0
            for iaX in range(-naX, naX + 1):
                # mean Etamx value for certain error (iaX) of parameter aVar
                aXmean[iaX + naX] = np.mean(aEtamx[iLDR, aX == iaX])
            # relative to absolute spread of Etamx
            meanDist = (np.max(aXmean) - np.min(aXmean)) / (Etamxmax - Etamxmin) * 100

            plt.subplot(1, 5, iLDR + 1)
            (n, bins, patches) = plt.hist(
                aEtamx[iLDR, :],
                bins=50,
                log=False,
                range=[Rmin, Rmax],
                alpha=0.5,
                density=False,
                color="0.5",
                histtype="stepfilled",
            )
            for iaX in range(-naX, naX + 1):
                plt.hist(
                    aEtamx[iLDR, aX == iaX],
                    range=[Rmin, Rmax],
                    bins=50,
                    log=False,
                    alpha=0.3,
                    density=False,
                    histtype="stepfilled",
                    label=str(round(X0 + iaX * daX / naX, 5)),
                )
                if iLDR == 2:
                    leg = plt.legend()
                    leg.get_frame().set_alpha(0.1)
            plt.tick_params(axis="both", labelsize=10)
            plt.plot([Etamx0, Etamx0], [0, np.max(n)], "r-", lw=2)
            plt.gca().set_title("{0:3.0f}%".format(meanDist))
            plt.gca().set_xlabel("Etamx0", color="red")
        fig.suptitle(
            "Etamx - "
            + LID
            + " with "
            + str(Type[TypeC])
            + " "
            + str(Loc[LocC])
            + " - "
            + aVar
            + " error contribution",
            fontsize=14,
            y=1.10,
        )
        return

    # calc contribution of the error of aVar = aX  to aY for each LDRtrue
    def Contribution(aVar, aX, X0, daX, iaX, naX, aY, Ysum, widthSum):
        # aVar is the name of the parameter and aX is the subset of aY which is coloured in the plot
        # example: Contribution("DOLP", aDOLP, DOLP0, dDOLP, iDOLP, nDOLP, aLDRcorr, DOLPcontr)
        iLDR = -1
        # Ysum, widthSum = np.zeros(5)
        meanDist = np.zeros(5)  # iLDR
        widthDist = np.zeros(5)  # iLDR
        for LDRTrue in LDRrange:
            aXmean = np.zeros(2 * naX + 1)
            aXwidth = np.zeros(2 * naX + 1)
            iLDR = iLDR + 1
            # total width of distribution
            aYmin = np.amin(aY[iLDR, :])
            aYmax = np.amax(aY[iLDR, :])
            aYwidth = aYmax - aYmin
            # Determine mean distance of all aXmean from each other for each iLDR
            for iaX in range(-naX, naX + 1):
                # mean LDRCorr value for all errors iaX of parameter aVar
                aXmean[iaX + naX] = np.mean(aY[iLDR, aX == iaX])
                aXwidth[iaX + naX] = np.max(aY[iLDR, aX == iaX]) - np.min(
                    aY[iLDR, aX == iaX]
                )
            # relative to absolute spread of LDRCorrs
            meanDist[iLDR] = (np.max(aXmean) - np.min(aXmean)) / aYwidth * 1000
            # meanDist[iLDR] = (aYwidth - aXwidth[naX]) / aYwidth * 1000
            widthDist[iLDR] = (np.max(aXwidth) - aXwidth[naX]) / aYwidth * 1000

        print(
            "{:12}{:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f}    {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f}".format(
                aVar,
                meanDist[0],
                meanDist[1],
                meanDist[2],
                meanDist[3],
                meanDist[4],
                widthDist[0],
                widthDist[1],
                widthDist[2],
                widthDist[3],
                widthDist[4],
            )
        )
        Ysum = Ysum + meanDist
        widthSum = widthSum + widthDist
        return (Ysum, widthSum)

        # print(.format(LDRrangeA[iLDR],))

    # error contributions to a certain output aY; loop over all variables
    def Contribution_aY(aYvar, aY):
        Ysum = np.zeros(5)
        widthSum = np.zeros(5)
        # meanDist = np.zeros(5) # iLDR
        LDRrangeA = np.array(LDRrange)
        print()
        print(aYvar + ": contribution to the total error (per mill)")
        print(
            "          of individual parameter errors        of combined parameter errors"
        )
        print(
            " at LDRtrue {:5.3f} {:5.3f} {:5.3f} {:5.3f} {:5.3f}    {:5.3f} {:5.3f} {:5.3f} {:5.3f} {:5.3f}".format(
                LDRrangeA[0],
                LDRrangeA[1],
                LDRrangeA[2],
                LDRrangeA[3],
                LDRrangeA[4],
                LDRrangeA[0],
                LDRrangeA[1],
                LDRrangeA[2],
                LDRrangeA[3],
                LDRrangeA[4],
            )
        )
        print()
        if nQin > 0:
            Ysum, widthSum = Contribution(
                "Qin", aQin, Qin0, dQin, iQin, nQin, aY, Ysum, widthSum
            )
        if nVin > 0:
            Ysum, widthSum = Contribution(
                "Vin", aVin, Vin0, dVin, iVin, nVin, aY, Ysum, widthSum
            )
        if nRotL > 0:
            Ysum, widthSum = Contribution(
                "RotL", aRotL, RotL0, dRotL, iRotL, nRotL, aY, Ysum, widthSum
            )
        if nRetE > 0:
            Ysum, widthSum = Contribution(
                "RetE", aRetE, RetE0, dRetE, iRetE, nRetE, aY, Ysum, widthSum
            )
        if nRotE > 0:
            Ysum, widthSum = Contribution(
                "RotE", aRotE, RotE0, dRotE, iRotE, nRotE, aY, Ysum, widthSum
            )
        if nDiE > 0:
            Ysum, widthSum = Contribution(
                "DiE", aDiE, DiE0, dDiE, iDiE, nDiE, aY, Ysum, widthSum
            )
        if nRetO > 0:
            Ysum, widthSum = Contribution(
                "RetO", aRetO, RetO0, dRetO, iRetO, nRetO, aY, Ysum, widthSum
            )
        if nRotO > 0:
            Ysum, widthSum = Contribution(
                "RotO", aRotO, RotO0, dRotO, iRotO, nRotO, aY, Ysum, widthSum
            )
        if nDiO > 0:
            Ysum, widthSum = Contribution(
                "DiO", aDiO, DiO0, dDiO, iDiO, nDiO, aY, Ysum, widthSum
            )
        if nDiC > 0:
            Ysum, widthSum = Contribution(
                "DiC", aDiC, DiC0, dDiC, iDiC, nDiC, aY, Ysum, widthSum
            )
        if nRotC > 0:
            Ysum, widthSum = Contribution(
                "RotC", aRotC, RotC0, dRotC, iRotC, nRotC, aY, Ysum, widthSum
            )
        if nRetC > 0:
            Ysum, widthSum = Contribution(
                "RetC", aRetC, RetC0, dRetC, iRetC, nRetC, aY, Ysum, widthSum
            )
        if nTP > 0:
            Ysum, widthSum = Contribution(
                "TP", aTP, TP0, dTP, iTP, nTP, aY, Ysum, widthSum
            )
        if nTS > 0:
            Ysum, widthSum = Contribution(
                "TS", aTS, TS0, dTS, iTS, nTS, aY, Ysum, widthSum
            )
        if nRP > 0:
            Ysum, widthSum = Contribution(
                "RP", aRP, RP0, dRP, iRP, nRP, aY, Ysum, widthSum
            )
        if nRS > 0:
            Ysum, widthSum = Contribution(
                "RS", aRS, RS0, dRS, iRS, nRS, aY, Ysum, widthSum
            )
        if nRetT > 0:
            Ysum, widthSum = Contribution(
                "RetT", aRetT, RetT0, dRetT, iRetT, nRetT, aY, Ysum, widthSum
            )
        if nRetR > 0:
            Ysum, widthSum = Contribution(
                "RetR", aRetR, RetR0, dRetR, iRetR, nRetR, aY, Ysum, widthSum
            )
        if nERaT > 0:
            Ysum, widthSum = Contribution(
                "ERaT", aERaT, ERaT0, dERaT, iERaT, nERaT, aY, Ysum, widthSum
            )
        if nERaR > 0:
            Ysum, widthSum = Contribution(
                "ERaR", aERaR, ERaR0, dERaR, iERaR, nERaR, aY, Ysum, widthSum
            )
        if nRotaT > 0:
            Ysum, widthSum = Contribution(
                "RotaT", aRotaT, RotaT0, dRotaT, iRotaT, nRotaT, aY, Ysum, widthSum
            )
        if nRotaR > 0:
            Ysum, widthSum = Contribution(
                "RotaR", aRotaR, RotaR0, dRotaR, iRotaR, nRotaR, aY, Ysum, widthSum
            )
        if nLDRCal > 0:
            Ysum, widthSum = Contribution(
                "LDRCal",
                aLDRCal,
                LDRCal0,
                dLDRCal,
                iLDRCal,
                nLDRCal,
                aY,
                Ysum,
                widthSum,
            )
        if nTCalT > 0:
            Ysum, widthSum = Contribution(
                "TCalT", aTCalT, TCalT0, dTCalT, iTCalT, nTCalT, aY, Ysum, widthSum
            )
        if nTCalR > 0:
            Ysum, widthSum = Contribution(
                "TCalR", aTCalR, TCalR0, dTCalR, iTCalR, nTCalR, aY, Ysum, widthSum
            )
        if nNCal > 0:
            Ysum, widthSum = Contribution(
                "CalNoiseTp", aNCalTp, 0, 1, iNCalTp, nNCal, aY, Ysum, widthSum
            )
        if nNCal > 0:
            Ysum, widthSum = Contribution(
                "CalNoiseTm", aNCalTm, 0, 1, iNCalTm, nNCal, aY, Ysum, widthSum
            )
        if nNCal > 0:
            Ysum, widthSum = Contribution(
                "CalNoiseRp", aNCalRp, 0, 1, iNCalRp, nNCal, aY, Ysum, widthSum
            )
        if nNCal > 0:
            Ysum, widthSum = Contribution(
                "CalNoiseRm", aNCalRm, 0, 1, iNCalRm, nNCal, aY, Ysum, widthSum
            )
        if nNI > 0:
            Ysum, widthSum = Contribution(
                "SigNoiseIt", aNIt, 0, 1, iNIt, nNI, aY, Ysum, widthSum
            )
        if nNI > 0:
            Ysum, widthSum = Contribution(
                "SigNoiseIr", aNIr, 0, 1, iNIr, nNI, aY, Ysum, widthSum
            )
        print(
            "{:12}{:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f}    {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f}".format(
                "Sum ",
                Ysum[0],
                Ysum[1],
                Ysum[2],
                Ysum[3],
                Ysum[4],
                widthSum[0],
                widthSum[1],
                widthSum[2],
                widthSum[3],
                widthSum[4],
            )
        )

    # Plot LDR histograms
    if nQin > 0:
        PlotSubHist("Qin", aQin, Qin0, dQin, iQin, nQin)
    if nVin > 0:
        PlotSubHist("Vin", aVin, Vin0, dVin, iVin, nVin)
    if nRotL > 0:
        PlotSubHist("RotL", aRotL, RotL0, dRotL, iRotL, nRotL)
    if nRetE > 0:
        PlotSubHist("RetE", aRetE, RetE0, dRetE, iRetE, nRetE)
    if nRotE > 0:
        PlotSubHist("RotE", aRotE, RotE0, dRotE, iRotE, nRotE)
    if nDiE > 0:
        PlotSubHist("DiE", aDiE, DiE0, dDiE, iDiE, nDiE)
    if nRetO > 0:
        PlotSubHist("RetO", aRetO, RetO0, dRetO, iRetO, nRetO)
    if nRotO > 0:
        PlotSubHist("RotO", aRotO, RotO0, dRotO, iRotO, nRotO)
    if nDiO > 0:
        PlotSubHist("DiO", aDiO, DiO0, dDiO, iDiO, nDiO)
    if nDiC > 0:
        PlotSubHist("DiC", aDiC, DiC0, dDiC, iDiC, nDiC)
    if nRotC > 0:
        PlotSubHist("RotC", aRotC, RotC0, dRotC, iRotC, nRotC)
    if nRetC > 0:
        PlotSubHist("RetC", aRetC, RetC0, dRetC, iRetC, nRetC)
    if nTP > 0:
        PlotSubHist("TP", aTP, TP0, dTP, iTP, nTP)
    if nTS > 0:
        PlotSubHist("TS", aTS, TS0, dTS, iTS, nTS)
    if nRP > 0:
        PlotSubHist("RP", aRP, RP0, dRP, iRP, nRP)
    if nRS > 0:
        PlotSubHist("RS", aRS, RS0, dRS, iRS, nRS)
    if nRetT > 0:
        PlotSubHist("RetT", aRetT, RetT0, dRetT, iRetT, nRetT)
    if nRetR > 0:
        PlotSubHist("RetR", aRetR, RetR0, dRetR, iRetR, nRetR)
    if nERaT > 0:
        PlotSubHist("ERaT", aERaT, ERaT0, dERaT, iERaT, nERaT)
    if nERaR > 0:
        PlotSubHist("ERaR", aERaR, ERaR0, dERaR, iERaR, nERaR)
    if nRotaT > 0:
        PlotSubHist("RotaT", aRotaT, RotaT0, dRotaT, iRotaT, nRotaT)
    if nRotaR > 0:
        PlotSubHist("RotaR", aRotaR, RotaR0, dRotaR, iRotaR, nRotaR)
    if nLDRCal > 0:
        PlotSubHist("LDRCal", aLDRCal, LDRCal0, dLDRCal, iLDRCal, nLDRCal)
    if nTCalT > 0:
        PlotSubHist("TCalT", aTCalT, TCalT0, dTCalT, iTCalT, nTCalT)
    if nTCalR > 0:
        PlotSubHist("TCalR", aTCalR, TCalR0, dTCalR, iTCalR, nTCalR)
    if nNCal > 0:
        PlotSubHist("CalNoiseTp", aNCalTp, 0, 1, iNCalTp, nNCal)
    if nNCal > 0:
        PlotSubHist("CalNoiseTm", aNCalTm, 0, 1, iNCalTm, nNCal)
    if nNCal > 0:
        PlotSubHist("CalNoiseRp", aNCalRp, 0, 1, iNCalRp, nNCal)
    if nNCal > 0:
        PlotSubHist("CalNoiseRm", aNCalRm, 0, 1, iNCalRm, nNCal)
    if nNI > 0:
        PlotSubHist("SigNoiseIt", aNIt, 0, 1, iNIt, nNI)
    if nNI > 0:
        PlotSubHist("SigNoiseIr", aNIr, 0, 1, iNIr, nNI)
    plt.show()
    plt.close

    # --- Plot LDRmin, LDRmax
    iLDR = -1
    for LDRTrue in LDRrange:
        iLDR = iLDR + 1
        LDRmin[iLDR] = np.amin(aLDRcorr[iLDR, :])
        LDRmax[iLDR] = np.amax(aLDRcorr[iLDR, :])
        LDRstd[iLDR] = np.std(aLDRcorr[iLDR, :])
        LDRmean[iLDR] = np.mean(aLDRcorr[iLDR, :])
        LDRmedian[iLDR] = np.median(aLDRcorr[iLDR, :])
        LDRskew[iLDR] = skew(aLDRcorr[iLDR, :], bias=False)
        LDRkurt[iLDR] = kurtosis(aLDRcorr[iLDR, :], fisher=True, bias=False)

    fig2 = plt.figure()
    LDRrangeA = np.array(LDRrange)
    if (np.amax(LDRmax - LDRrangeA) - np.amin(LDRmin - LDRrangeA)) < 0.001:
        plt.ylim(-0.001, 0.001)
    plt.plot(LDRrangeA, LDRmax - LDRrangeA, linewidth=2.0, color="b")
    plt.plot(LDRrangeA, LDRmin - LDRrangeA, linewidth=2.0, color="g")

    plt.xlabel("LDRtrue", fontsize=18)
    plt.ylabel("LDR: min - true, max - true", fontsize=14)
    plt.title(LID + " " + str(Type[TypeC]) + " " + str(Loc[LocC]), fontsize=18)
    # plt.ylimit(-0.07, 0.07)
    plt.show()
    plt.close

    # Print LDR statistics
    print(
        "LDRtrue ,  mean  ,  median,    max-true,  min-true, std,   excess_kurtosis, skewness"
    )
    iLDR = -1
    LDRrangeA = np.array(LDRrange)
    for LDRTrue in LDRrange:
        iLDR = iLDR + 1
        print(
            "{0:8.5f},{1:8.5f},{2:8.5f},    {3:8.5f},{4:8.5f},{5:8.5f},   {6:8.5f},{7:8.5f}".format(
                LDRrangeA[iLDR],
                LDRmean[iLDR],
                LDRmedian[iLDR],
                LDRmax[iLDR] - LDRrangeA[iLDR],
                LDRmin[iLDR] - LDRrangeA[iLDR],
                LDRstd[iLDR],
                LDRkurt[iLDR],
                LDRskew[iLDR],
            )
        )
    print()
    # --- Calculate 2nd order polynom fit to lower and upper LDR errors between LDR = 0.004 and 0.45 and also 2nd order fit to K(LDRcal)
    LDRmin_poly = np.polyfit(LDRrangeA, (LDRmin - LDRrangeA), 2)
    LDRmax_poly = np.polyfit(LDRrangeA, (LDRmax - LDRrangeA), 2)
    print(
        "LDRCal is the volume linear depolarisation ratio in the range used for the polarisation calibration. K(LDRCal) is the variability of K depending on LDRCal."
    )
    print(
        "If LDRCal is not known, its uncertainty has to be included in the input file for the error calcualtion below."
    )
    print(
        "LDR_min-true(LDR_true) and LDR_max-true(LDR_true) are the difference between the lowest and highest possible retrieved LDR, respectively, and the true LDR depending on the true LDR. They represent the negative and positive uncertainty of the retrieved LDR resulting from all known uncertainties of the optical input parameters."
    )
    print("2nd order polyfit = a + (b * LDR) + (c * LDR^2)")
    print()
    print("K(LDRCal), LDR_min-true(LDRtrue), LDR_max-true(LDRtrue)")
    print(
        " a, ",
        f"{K_poly[2]:+.5f},",
        f"{LDRmin_poly[2]:+.5f},",
        f"{LDRmax_poly[2]:+.5f}",
    )
    print(
        " b, ",
        f"{K_poly[1]:+.5f},",
        f"{LDRmin_poly[1]:+.5f},",
        f"{LDRmax_poly[1]:+.5f}",
    )
    print(
        " c, ",
        f"{K_poly[0]:+.5f},",
        f"{LDRmin_poly[0]:+.5f},",
        f"{LDRmax_poly[0]:+.5f}",
    )
    print()

    # --- Save LDRmin, LDRmax to file
    # http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python

    with open(output_path, "a") as f:
        # with open('output_files\\' + LID + '-' + InputFile[0:-3] + '-LDR_min_max.dat', 'w') as f:
        with redirect_stdout(f):
            print("Lidar ID: " + LID)
            print()
            print("Correction K(LDRCal in calibration range)")
            print("LDRCal, K(LDRCal)")
            for i in range(1, 7):
                # print("{0:5.3f},{1:9.5f},{2:9.5f}".format(LDRCalList[i], K0List[i]), p(LDRCalList[i]))
                print("{0:5.3f},{1:9.5f}".format(LDRCalList[i], K0List[i]))
            print()
            print(
                "minimum and maximum values of the distributions of possibly measured LDR for different LDRtrue"
            )
            print("LDRtrue  , LDRmin, LDRmax")
            for i in range(len(LDRrangeA)):
                print(
                    "{0:7.4f},{1:7.4f},{2:7.4f}".format(
                        LDRrangeA[i], LDRmin[i], LDRmax[i]
                    )
                )
            print()
            print(
                "LDRCal is the volume linear depolarisation ratio in the range used for the polarisation calibration. K(LDRCal) is the variability of K depending on LDRCal."
            )
            print(
                "If LDRCal is not known, its uncertainty has to be included in the input file for the error calcualtion below."
            )
            print(
                "LDR_min-true(LDR_true) and LDR_max-true(LDR_true) are the difference between the lowest and highest possible retrieved LDR, respectively, and the true LDR depending on the true LDR. They represent the negative and positive uncertainty of the retrieved LDR resulting from all known uncertainties of the optical input parameters."
            )
            print("-----------------------------------------------")
            print("2nd order polyfit = a + (b * LDR) + (c * LDR^2)")
            print(
                "fit-parameter, K(LDRCal), LDR_min-true(LDRtrue), LDR_max-true(LDRtrue)"
            )
            print(
                " a, ",
                f"{K_poly[2]:+.5f},",
                f"{LDRmin_poly[2]:+.5f},",
                f"{LDRmax_poly[2]:+.5f}",
            )
            print(
                " b, ",
                f"{K_poly[1]:+.5f},",
                f"{LDRmin_poly[1]:+.5f},",
                f"{LDRmax_poly[1]:+.5f}",
            )
            print(
                " c, ",
                f"{K_poly[0]:+.5f},",
                f"{LDRmin_poly[0]:+.5f},",
                f"{LDRmax_poly[0]:+.5f}",
            )
            print()
            # Print LDR statistics
            print(
                "##################################################################################"
            )
            print(
                "Further statistical analyses of the uncertainties, their distribution and sources:"
            )
            print()
            print(
                "LDRtrue ,  mean  ,  median,    max-mean,  min-mean, std,   excess_kurtosis, skewness"
            )
            iLDR = -1
            LDRrangeA = np.array(LDRrange)
            for LDRTrue in LDRrange:
                iLDR = iLDR + 1
                print(
                    "{0:8.5f},{1:8.5f},{2:8.5f},    {3:8.5f},{4:8.5f},{5:8.5f},   {6:8.5f},{7:8.5f}".format(
                        LDRrangeA[iLDR],
                        LDRmean[iLDR],
                        LDRmedian[iLDR],
                        LDRmax[iLDR] - LDRrangeA[iLDR],
                        LDRmin[iLDR] - LDRrangeA[iLDR],
                        LDRstd[iLDR],
                        LDRkurt[iLDR],
                        LDRskew[iLDR],
                    )
                )
            print()
            # Calculate and print statistics for calibration factors
            print(
                "minimum and maximum values of the distributions of signal ratios and calibration factors for different LDRtrue"
            )
            iLDR = -1
            LDRrangeA = np.array(LDRrange)
            print("LDRtrue  , LDRsim, (max-min)/2, relerr")
            for LDRTrue in LDRrange:
                iLDR = iLDR + 1
                LDRsimmin[iLDR] = np.amin(aLDRsim[iLDR, :])
                LDRsimmax[iLDR] = np.amax(aLDRsim[iLDR, :])
                # LDRsimstd = np.std(aLDRsim[iLDR, :])
                LDRsimmean[iLDR] = np.mean(aLDRsim[iLDR, :])
                # LDRsimmedian = np.median(aLDRsim[iLDR, :])
                print(
                    "{0:8.5f}, {1:8.5f}, {2:8.5f}, {3:8.5f}".format(
                        LDRrangeA[iLDR],
                        LDRsimmean[iLDR],
                        (LDRsimmax[iLDR] - LDRsimmin[iLDR]) / 2,
                        (LDRsimmax[iLDR] - LDRsimmin[iLDR]) / 2 / LDRsimmean[iLDR],
                    )
                )
            iLDR = -1
            print("LDRtrue  , Etax   , (max-min)/2, relerr")
            for LDRTrue in LDRrange:
                iLDR = iLDR + 1
                Etaxmin = np.amin(aEtax[iLDR, :])
                Etaxmax = np.amax(aEtax[iLDR, :])
                # Etaxstd = np.std(aEtax[iLDR, :])
                Etaxmean = np.mean(aEtax[iLDR, :])
                # Etaxmedian = np.median(aEtax[iLDR, :])
                print(
                    "{0:8.5f}, {1:8.5f}, {2:8.5f}, {3:8.5f}".format(
                        LDRrangeA[iLDR],
                        Etaxmean,
                        (Etaxmax - Etaxmin) / 2,
                        (Etaxmax - Etaxmin) / 2 / Etaxmean,
                    )
                )
            iLDR = -1
            print("LDRtrue  , Etapx  , (max-min)/2, relerr")
            for LDRTrue in LDRrange:
                iLDR = iLDR + 1
                Etapxmin = np.amin(aEtapx[iLDR, :])
                Etapxmax = np.amax(aEtapx[iLDR, :])
                # Etapxstd = np.std(aEtapx[iLDR, :])
                Etapxmean = np.mean(aEtapx[iLDR, :])
                # Etapxmedian = np.median(aEtapx[iLDR, :])
                print(
                    "{0:8.5f}, {1:8.5f}, {2:8.5f}, {3:8.5f}".format(
                        LDRrangeA[iLDR],
                        Etapxmean,
                        (Etapxmax - Etapxmin) / 2,
                        (Etapxmax - Etapxmin) / 2 / Etapxmean,
                    )
                )
            iLDR = -1
            print("LDRtrue  , Etamx  , (max-min)/2, relerr")
            for LDRTrue in LDRrange:
                iLDR = iLDR + 1
                Etamxmin = np.amin(aEtamx[iLDR, :])
                Etamxmax = np.amax(aEtamx[iLDR, :])
                # Etamxstd = np.std(aEtamx[iLDR, :])
                Etamxmean = np.mean(aEtamx[iLDR, :])
                # Etamxmedian = np.median(aEtamx[iLDR, :])
                print(
                    "{0:8.5f}, {1:8.5f}, {2:8.5f}, {3:8.5f}".format(
                        LDRrangeA[iLDR],
                        Etamxmean,
                        (Etamxmax - Etamxmin) / 2,
                        (Etamxmax - Etamxmin) / 2 / Etamxmean,
                    )
                )

    with open(output_path, "a") as f:
        # with open('output_files\\' + LID + '-' + InputFile[0:-3] + '-LDR_min_max.dat', 'a') as f:
        with redirect_stdout(f):
            Contribution_aY("LDRCorr", aLDRcorr)
            Contribution_aY("LDRsim", aLDRsim)
            Contribution_aY("EtaX, D90", aEtax)
            Contribution_aY("Etapx, +45°", aEtapx)
            Contribution_aY("Etamx -45°", aEtamx)

    # Plot other histograms
    if bPlotEtax:

        if nQin > 0:
            PlotLDRsim("Qin", aQin, Qin0, dQin, iQin, nQin)
        if nVin > 0:
            PlotLDRsim("Vin", aVin, Vin0, dVin, iVin, nVin)
        if nRotL > 0:
            PlotLDRsim("RotL", aRotL, RotL0, dRotL, iRotL, nRotL)
        if nRetE > 0:
            PlotLDRsim("RetE", aRetE, RetE0, dRetE, iRetE, nRetE)
        if nRotE > 0:
            PlotLDRsim("RotE", aRotE, RotE0, dRotE, iRotE, nRotE)
        if nDiE > 0:
            PlotLDRsim("DiE", aDiE, DiE0, dDiE, iDiE, nDiE)
        if nRetO > 0:
            PlotLDRsim("RetO", aRetO, RetO0, dRetO, iRetO, nRetO)
        if nRotO > 0:
            PlotLDRsim("RotO", aRotO, RotO0, dRotO, iRotO, nRotO)
        if nDiO > 0:
            PlotLDRsim("DiO", aDiO, DiO0, dDiO, iDiO, nDiO)
        if nDiC > 0:
            PlotLDRsim("DiC", aDiC, DiC0, dDiC, iDiC, nDiC)
        if nRotC > 0:
            PlotLDRsim("RotC", aRotC, RotC0, dRotC, iRotC, nRotC)
        if nRetC > 0:
            PlotLDRsim("RetC", aRetC, RetC0, dRetC, iRetC, nRetC)
        if nTP > 0:
            PlotLDRsim("TP", aTP, TP0, dTP, iTP, nTP)
        if nTS > 0:
            PlotLDRsim("TS", aTS, TS0, dTS, iTS, nTS)
        if nRP > 0:
            PlotLDRsim("RP", aRP, RP0, dRP, iRP, nRP)
        if nRS > 0:
            PlotLDRsim("RS", aRS, RS0, dRS, iRS, nRS)
        if nRetT > 0:
            PlotLDRsim("RetT", aRetT, RetT0, dRetT, iRetT, nRetT)
        if nRetR > 0:
            PlotLDRsim("RetR", aRetR, RetR0, dRetR, iRetR, nRetR)
        if nERaT > 0:
            PlotLDRsim("ERaT", aERaT, ERaT0, dERaT, iERaT, nERaT)
        if nERaR > 0:
            PlotLDRsim("ERaR", aERaR, ERaR0, dERaR, iERaR, nERaR)
        if nRotaT > 0:
            PlotLDRsim("RotaT", aRotaT, RotaT0, dRotaT, iRotaT, nRotaT)
        if nRotaR > 0:
            PlotLDRsim("RotaR", aRotaR, RotaR0, dRotaR, iRotaR, nRotaR)
        if nLDRCal > 0:
            PlotLDRsim("LDRCal", aLDRCal, LDRCal0, dLDRCal, iLDRCal, nLDRCal)
        if nTCalT > 0:
            PlotLDRsim("TCalT", aTCalT, TCalT0, dTCalT, iTCalT, nTCalT)
        if nTCalR > 0:
            PlotLDRsim("TCalR", aTCalR, TCalR0, dTCalR, iTCalR, nTCalR)
        if nNCal > 0:
            PlotLDRsim("CalNoiseTp", aNCalTp, 0, 1, iNCalTp, nNCal)
        if nNCal > 0:
            PlotLDRsim("CalNoiseTm", aNCalTm, 0, 1, iNCalTm, nNCal)
        if nNCal > 0:
            PlotLDRsim("CalNoiseRp", aNCalRp, 0, 1, iNCalRp, nNCal)
        if nNCal > 0:
            PlotLDRsim("CalNoiseRm", aNCalRm, 0, 1, iNCalRm, nNCal)
        if nNI > 0:
            PlotLDRsim("SigNoiseIt", aNIt, 0, 1, iNIt, nNI)
        if nNI > 0:
            PlotLDRsim("SigNoiseIr", aNIr, 0, 1, iNIr, nNI)
        plt.show()
        plt.close
        print(
            "---------------------------------------...producing more plots...------------------------------------------------------------------"
        )

        if nQin > 0:
            PlotEtax("Qin", aQin, Qin0, dQin, iQin, nQin)
        if nVin > 0:
            PlotEtax("Vin", aVin, Vin0, dVin, iVin, nVin)
        if nRotL > 0:
            PlotEtax("RotL", aRotL, RotL0, dRotL, iRotL, nRotL)
        if nRetE > 0:
            PlotEtax("RetE", aRetE, RetE0, dRetE, iRetE, nRetE)
        if nRotE > 0:
            PlotEtax("RotE", aRotE, RotE0, dRotE, iRotE, nRotE)
        if nDiE > 0:
            PlotEtax("DiE", aDiE, DiE0, dDiE, iDiE, nDiE)
        if nRetO > 0:
            PlotEtax("RetO", aRetO, RetO0, dRetO, iRetO, nRetO)
        if nRotO > 0:
            PlotEtax("RotO", aRotO, RotO0, dRotO, iRotO, nRotO)
        if nDiO > 0:
            PlotEtax("DiO", aDiO, DiO0, dDiO, iDiO, nDiO)
        if nDiC > 0:
            PlotEtax("DiC", aDiC, DiC0, dDiC, iDiC, nDiC)
        if nRotC > 0:
            PlotEtax("RotC", aRotC, RotC0, dRotC, iRotC, nRotC)
        if nRetC > 0:
            PlotEtax("RetC", aRetC, RetC0, dRetC, iRetC, nRetC)
        if nTP > 0:
            PlotEtax("TP", aTP, TP0, dTP, iTP, nTP)
        if nTS > 0:
            PlotEtax("TS", aTS, TS0, dTS, iTS, nTS)
        if nRP > 0:
            PlotEtax("RP", aRP, RP0, dRP, iRP, nRP)
        if nRS > 0:
            PlotEtax("RS", aRS, RS0, dRS, iRS, nRS)
        if nRetT > 0:
            PlotEtax("RetT", aRetT, RetT0, dRetT, iRetT, nRetT)
        if nRetR > 0:
            PlotEtax("RetR", aRetR, RetR0, dRetR, iRetR, nRetR)
        if nERaT > 0:
            PlotEtax("ERaT", aERaT, ERaT0, dERaT, iERaT, nERaT)
        if nERaR > 0:
            PlotEtax("ERaR", aERaR, ERaR0, dERaR, iERaR, nERaR)
        if nRotaT > 0:
            PlotEtax("RotaT", aRotaT, RotaT0, dRotaT, iRotaT, nRotaT)
        if nRotaR > 0:
            PlotEtax("RotaR", aRotaR, RotaR0, dRotaR, iRotaR, nRotaR)
        if nLDRCal > 0:
            PlotEtax("LDRCal", aLDRCal, LDRCal0, dLDRCal, iLDRCal, nLDRCal)
        if nTCalT > 0:
            PlotEtax("TCalT", aTCalT, TCalT0, dTCalT, iTCalT, nTCalT)
        if nTCalR > 0:
            PlotEtax("TCalR", aTCalR, TCalR0, dTCalR, iTCalR, nTCalR)
        if nNCal > 0:
            PlotEtax("CalNoiseTp", aNCalTp, 0, 1, iNCalTp, nNCal)
        if nNCal > 0:
            PlotEtax("CalNoiseTm", aNCalTm, 0, 1, iNCalTm, nNCal)
        if nNCal > 0:
            PlotEtax("CalNoiseRp", aNCalRp, 0, 1, iNCalRp, nNCal)
        if nNCal > 0:
            PlotEtax("CalNoiseRm", aNCalRm, 0, 1, iNCalRm, nNCal)
        if nNI > 0:
            PlotEtax("SigNoiseIt", aNIt, 0, 1, iNIt, nNI)
        if nNI > 0:
            PlotEtax("SigNoiseIr", aNIr, 0, 1, iNIr, nNI)
        plt.show()
        plt.close
        print(
            "---------------------------------------...producing more plots...------------------------------------------------------------------"
        )

        if nQin > 0:
            PlotEtapx("Qin", aQin, Qin0, dQin, iQin, nQin)
        if nVin > 0:
            PlotEtapx("Vin", aVin, Vin0, dVin, iVin, nVin)
        if nRotL > 0:
            PlotEtapx("RotL", aRotL, RotL0, dRotL, iRotL, nRotL)
        if nRetE > 0:
            PlotEtapx("RetE", aRetE, RetE0, dRetE, iRetE, nRetE)
        if nRotE > 0:
            PlotEtapx("RotE", aRotE, RotE0, dRotE, iRotE, nRotE)
        if nDiE > 0:
            PlotEtapx("DiE", aDiE, DiE0, dDiE, iDiE, nDiE)
        if nRetO > 0:
            PlotEtapx("RetO", aRetO, RetO0, dRetO, iRetO, nRetO)
        if nRotO > 0:
            PlotEtapx("RotO", aRotO, RotO0, dRotO, iRotO, nRotO)
        if nDiO > 0:
            PlotEtapx("DiO", aDiO, DiO0, dDiO, iDiO, nDiO)
        if nDiC > 0:
            PlotEtapx("DiC", aDiC, DiC0, dDiC, iDiC, nDiC)
        if nRotC > 0:
            PlotEtapx("RotC", aRotC, RotC0, dRotC, iRotC, nRotC)
        if nRetC > 0:
            PlotEtapx("RetC", aRetC, RetC0, dRetC, iRetC, nRetC)
        if nTP > 0:
            PlotEtapx("TP", aTP, TP0, dTP, iTP, nTP)
        if nTS > 0:
            PlotEtapx("TS", aTS, TS0, dTS, iTS, nTS)
        if nRP > 0:
            PlotEtapx("RP", aRP, RP0, dRP, iRP, nRP)
        if nRS > 0:
            PlotEtapx("RS", aRS, RS0, dRS, iRS, nRS)
        if nRetT > 0:
            PlotEtapx("RetT", aRetT, RetT0, dRetT, iRetT, nRetT)
        if nRetR > 0:
            PlotEtapx("RetR", aRetR, RetR0, dRetR, iRetR, nRetR)
        if nERaT > 0:
            PlotEtapx("ERaT", aERaT, ERaT0, dERaT, iERaT, nERaT)
        if nERaR > 0:
            PlotEtapx("ERaR", aERaR, ERaR0, dERaR, iERaR, nERaR)
        if nRotaT > 0:
            PlotEtapx("RotaT", aRotaT, RotaT0, dRotaT, iRotaT, nRotaT)
        if nRotaR > 0:
            PlotEtapx("RotaR", aRotaR, RotaR0, dRotaR, iRotaR, nRotaR)
        if nLDRCal > 0:
            PlotEtapx("LDRCal", aLDRCal, LDRCal0, dLDRCal, iLDRCal, nLDRCal)
        if nTCalT > 0:
            PlotEtapx("TCalT", aTCalT, TCalT0, dTCalT, iTCalT, nTCalT)
        if nTCalR > 0:
            PlotEtapx("TCalR", aTCalR, TCalR0, dTCalR, iTCalR, nTCalR)
        if nNCal > 0:
            PlotEtapx("CalNoiseTp", aNCalTp, 0, 1, iNCalTp, nNCal)
        if nNCal > 0:
            PlotEtapx("CalNoiseTm", aNCalTm, 0, 1, iNCalTm, nNCal)
        if nNCal > 0:
            PlotEtapx("CalNoiseRp", aNCalRp, 0, 1, iNCalRp, nNCal)
        if nNCal > 0:
            PlotEtapx("CalNoiseRm", aNCalRm, 0, 1, iNCalRm, nNCal)
        if nNI > 0:
            PlotEtapx("SigNoiseIt", aNIt, 0, 1, iNIt, nNI)
        if nNI > 0:
            PlotEtapx("SigNoiseIr", aNIr, 0, 1, iNIr, nNI)
        plt.show()
        plt.close
        print(
            "---------------------------------------...producing more plots...------------------------------------------------------------------"
        )

        if nQin > 0:
            PlotEtamx("Qin", aQin, Qin0, dQin, iQin, nQin)
        if nVin > 0:
            PlotEtamx("Vin", aVin, Vin0, dVin, iVin, nVin)
        if nRotL > 0:
            PlotEtamx("RotL", aRotL, RotL0, dRotL, iRotL, nRotL)
        if nRetE > 0:
            PlotEtamx("RetE", aRetE, RetE0, dRetE, iRetE, nRetE)
        if nRotE > 0:
            PlotEtamx("RotE", aRotE, RotE0, dRotE, iRotE, nRotE)
        if nDiE > 0:
            PlotEtamx("DiE", aDiE, DiE0, dDiE, iDiE, nDiE)
        if nRetO > 0:
            PlotEtamx("RetO", aRetO, RetO0, dRetO, iRetO, nRetO)
        if nRotO > 0:
            PlotEtamx("RotO", aRotO, RotO0, dRotO, iRotO, nRotO)
        if nDiO > 0:
            PlotEtamx("DiO", aDiO, DiO0, dDiO, iDiO, nDiO)
        if nDiC > 0:
            PlotEtamx("DiC", aDiC, DiC0, dDiC, iDiC, nDiC)
        if nRotC > 0:
            PlotEtamx("RotC", aRotC, RotC0, dRotC, iRotC, nRotC)
        if nRetC > 0:
            PlotEtamx("RetC", aRetC, RetC0, dRetC, iRetC, nRetC)
        if nTP > 0:
            PlotEtamx("TP", aTP, TP0, dTP, iTP, nTP)
        if nTS > 0:
            PlotEtamx("TS", aTS, TS0, dTS, iTS, nTS)
        if nRP > 0:
            PlotEtamx("RP", aRP, RP0, dRP, iRP, nRP)
        if nRS > 0:
            PlotEtamx("RS", aRS, RS0, dRS, iRS, nRS)
        if nRetT > 0:
            PlotEtamx("RetT", aRetT, RetT0, dRetT, iRetT, nRetT)
        if nRetR > 0:
            PlotEtamx("RetR", aRetR, RetR0, dRetR, iRetR, nRetR)
        if nERaT > 0:
            PlotEtamx("ERaT", aERaT, ERaT0, dERaT, iERaT, nERaT)
        if nERaR > 0:
            PlotEtamx("ERaR", aERaR, ERaR0, dERaR, iERaR, nERaR)
        if nRotaT > 0:
            PlotEtamx("RotaT", aRotaT, RotaT0, dRotaT, iRotaT, nRotaT)
        if nRotaR > 0:
            PlotEtamx("RotaR", aRotaR, RotaR0, dRotaR, iRotaR, nRotaR)
        if nLDRCal > 0:
            PlotEtamx("LDRCal", aLDRCal, LDRCal0, dLDRCal, iLDRCal, nLDRCal)
        if nTCalT > 0:
            PlotEtamx("TCalT", aTCalT, TCalT0, dTCalT, iTCalT, nTCalT)
        if nTCalR > 0:
            PlotEtamx("TCalR", aTCalR, TCalR0, dTCalR, iTCalR, nTCalR)
        if nNCal > 0:
            PlotEtamx("CalNoiseTp", aNCalTp, 0, 1, iNCalTp, nNCal)
        if nNCal > 0:
            PlotEtamx("CalNoiseTm", aNCalTm, 0, 1, iNCalTm, nNCal)
        if nNCal > 0:
            PlotEtamx("CalNoiseRp", aNCalRp, 0, 1, iNCalRp, nNCal)
        if nNCal > 0:
            PlotEtamx("CalNoiseRm", aNCalRm, 0, 1, iNCalRm, nNCal)
        if nNI > 0:
            PlotEtamx("SigNoiseIt", aNIt, 0, 1, iNIt, nNI)
        if nNI > 0:
            PlotEtamx("SigNoiseIr", aNIr, 0, 1, iNIr, nNI)
        plt.show()
        plt.close

        # Print Etax statistics
        Etaxmin = np.amin(aEtax[1, :])
        Etaxmax = np.amax(aEtax[1, :])
        Etaxstd = np.std(aEtax[1, :])
        Etaxmean = np.mean(aEtax[1, :])
        Etaxmedian = np.median(aEtax[1, :])
        print("Etax      , max-mean, min-mean, median, mean ± std, eta")
        print(
            "{0:8.5f} ±({1:8.5f},{2:8.5f}),{3:8.5f},{4:8.5f}±{5:8.5f},{6:8.5f}".format(
                Etax0,
                Etaxmax - Etax0,
                Etaxmin - Etax0,
                Etaxmedian,
                Etaxmean,
                Etaxstd,
                Etax0 / K0,
            )
        )
        print()

        # Calculate and print statistics for calibration factors
        iLDR = -1
        LDRrangeA = np.array(LDRrange)
        print("LDR...., LDRsim, (max-min)/2, relerr")
        for LDRTrue in LDRrange:
            iLDR = iLDR + 1
            LDRsimmin[iLDR] = np.amin(aLDRsim[iLDR, :])
            LDRsimmax[iLDR] = np.amax(aLDRsim[iLDR, :])
            # LDRsimstd = np.std(aLDRsim[iLDR, :])
            LDRsimmean[iLDR] = np.mean(aLDRsim[iLDR, :])
            # LDRsimmedian = np.median(aLDRsim[iLDR, :])
            print(
                "{0:8.5f}, {1:8.5f}, {2:8.5f}, {3:8.5f}".format(
                    LDRrangeA[iLDR],
                    LDRsimmean[iLDR],
                    (LDRsimmax[iLDR] - LDRsimmin[iLDR]) / 2,
                    (LDRsimmax[iLDR] - LDRsimmin[iLDR]) / 2 / LDRsimmean[iLDR],
                )
            )
        iLDR = -1
        print("LDR...., Etax   , (max-min)/2, relerr")
        for LDRTrue in LDRrange:
            iLDR = iLDR + 1
            Etaxmin = np.amin(aEtax[iLDR, :])
            Etaxmax = np.amax(aEtax[iLDR, :])
            # Etaxstd = np.std(aEtax[iLDR, :])
            Etaxmean = np.mean(aEtax[iLDR, :])
            # Etaxmedian = np.median(aEtax[iLDR, :])
            print(
                "{0:8.5f}, {1:8.5f}, {2:8.5f}, {3:8.5f}".format(
                    LDRrangeA[iLDR],
                    Etaxmean,
                    (Etaxmax - Etaxmin) / 2,
                    (Etaxmax - Etaxmin) / 2 / Etaxmean,
                )
            )
        iLDR = -1
        print("LDR...., Etapx  , (max-min)/2, relerr")
        for LDRTrue in LDRrange:
            iLDR = iLDR + 1
            Etapxmin = np.amin(aEtapx[iLDR, :])
            Etapxmax = np.amax(aEtapx[iLDR, :])
            # Etapxstd = np.std(aEtapx[iLDR, :])
            Etapxmean = np.mean(aEtapx[iLDR, :])
            # Etapxmedian = np.median(aEtapx[iLDR, :])
            print(
                "{0:8.5f}, {1:8.5f}, {2:8.5f}, {3:8.5f}".format(
                    LDRrangeA[iLDR],
                    Etapxmean,
                    (Etapxmax - Etapxmin) / 2,
                    (Etapxmax - Etapxmin) / 2 / Etapxmean,
                )
            )
        iLDR = -1
        print("LDR...., Etamx  , (max-min)/2, relerr")
        for LDRTrue in LDRrange:
            iLDR = iLDR + 1
            Etamxmin = np.amin(aEtamx[iLDR, :])
            Etamxmax = np.amax(aEtamx[iLDR, :])
            # Etamxstd = np.std(aEtamx[iLDR, :])
            Etamxmean = np.mean(aEtamx[iLDR, :])
            # Etamxmedian = np.median(aEtamx[iLDR, :])
            print(
                "{0:8.5f}, {1:8.5f}, {2:8.5f}, {3:8.5f}".format(
                    LDRrangeA[iLDR],
                    Etamxmean,
                    (Etamxmax - Etamxmin) / 2,
                    (Etamxmax - Etamxmin) / 2 / Etamxmean,
                )
            )

    f.close()


"""
    # --- Plot F11 histograms
    print()
    print(" ############################################################################## ")
    print(Text1)
    print()

    iLDR = 5
    for LDRTrue in LDRrange:
        iLDR = iLDR - 1
        #aF11corr[iLDR,:] = aF11corr[iLDR,:] / aF11corr[0,:] - 1.0
        aF11corr[iLDR,:] = aF11corr[iLDR,:] / aF11sim0[iLDR] - 1.0
    # Plot F11
    def PlotSubHistF11(aVar, aX, X0, daX, iaX, naX):
        fig, ax = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(25, 2))
        iLDR = -1
        for LDRTrue in LDRrange:
            iLDR = iLDR + 1

            #F11min[iLDR] = np.min(aF11corr[iLDR,:])
            #F11max[iLDR] = np.max(aF11corr[iLDR,:])
            #Rmin = F11min[iLDR] * 0.995 #  np.min(aLDRcorr[iLDR,:])    * 0.995
            #Rmax = F11max[iLDR] * 1.005 #  np.max(aLDRcorr[iLDR,:])    * 1.005

            #Rmin = 0.8
            #Rmax = 1.2

            #plt.subplot(5,2,iLDR+1)
            plt.subplot(1,5,iLDR+1)
            (n, bins, patches) = plt.hist(aF11corr[iLDR,:],
                     bins=100, log=False,
                     alpha=0.5, density=False, color = '0.5', histtype='stepfilled')

            for iaX in range(-naX,naX+1):
                plt.hist(aF11corr[iLDR,aX == iaX],
                         bins=100, log=False, alpha=0.3, density=False, histtype='stepfilled', label = str(round(X0 + iaX*daX/naX,5)))

                if (iLDR == 2): plt.legend()

            plt.tick_params(axis='both', labelsize=9)
            #plt.plot([LDRTrue, LDRTrue], [0, np.max(n)], 'r-', lw=2)

        #plt.title(LID + '  ' + aVar, fontsize=18)
        #plt.ylabel('frequency', fontsize=10)
        #plt.xlabel('LDRCorr', fontsize=10)
        #fig.tight_layout()
        fig.suptitle(LID + '  ' + str(Type[TypeC]) + ' ' + str(Loc[LocC])  + ' - ' + aVar, fontsize=14, y=1.05)
        #plt.show()
        #fig.savefig(LID + '_' + aVar + '.png', dpi=150, bbox_inches='tight', pad_inches=0)
        #plt.close
        return

    if (nQin > 0): PlotSubHistF11("Qin", aQin, Qin0, dQin, iQin, nQin)
    if (nVin > 0): PlotSubHistF11("Vin", aVin, Vin0, dVin, iVin, nVin)
    if (nRotL > 0): PlotSubHistF11("RotL", aRotL, RotL0, dRotL, iRotL, nRotL)
    if (nRetE > 0): PlotSubHistF11("RetE", aRetE, RetE0, dRetE, iRetE, nRetE)
    if (nRotE > 0): PlotSubHistF11("RotE", aRotE, RotE0, dRotE, iRotE, nRotE)
    if (nDiE > 0): PlotSubHistF11("DiE", aDiE, DiE0, dDiE, iDiE, nDiE)
    if (nRetO > 0): PlotSubHistF11("RetO", aRetO, RetO0, dRetO, iRetO, nRetO)
    if (nRotO > 0): PlotSubHistF11("RotO", aRotO, RotO0, dRotO, iRotO, nRotO)
    if (nDiO > 0): PlotSubHistF11("DiO", aDiO, DiO0, dDiO, iDiO, nDiO)
    if (nDiC > 0): PlotSubHistF11("DiC", aDiC, DiC0, dDiC, iDiC, nDiC)
    if (nRotC > 0): PlotSubHistF11("RotC", aRotC, RotC0, dRotC, iRotC, nRotC)
    if (nRetC > 0): PlotSubHistF11("RetC", aRetC, RetC0, dRetC, iRetC, nRetC)
    if (nTP > 0): PlotSubHistF11("TP", aTP, TP0, dTP, iTP, nTP)
    if (nTS > 0): PlotSubHistF11("TS", aTS, TS0, dTS, iTS, nTS)
    if (nRP > 0): PlotSubHistF11("RP", aRP, RP0, dRP, iRP, nRP)
    if (nRS > 0): PlotSubHistF11("RS", aRS, RS0, dRS, iRS, nRS)
    if (nRetT > 0): PlotSubHistF11("RetT", aRetT, RetT0, dRetT, iRetT, nRetT)
    if (nRetR > 0): PlotSubHistF11("RetR", aRetR, RetR0, dRetR, iRetR, nRetR)
    if (nERaT > 0): PlotSubHistF11("ERaT", aERaT, ERaT0, dERaT, iERaT, nERaT)
    if (nERaR > 0): PlotSubHistF11("ERaR", aERaR, ERaR0, dERaR, iERaR, nERaR)
    if (nRotaT > 0): PlotSubHistF11("RotaT", aRotaT, RotaT0, dRotaT, iRotaT, nRotaT)
    if (nRotaR > 0): PlotSubHistF11("RotaR", aRotaR, RotaR0, dRotaR, iRotaR, nRotaR)
    if (nLDRCal > 0): PlotSubHistF11("LDRCal", aLDRCal, LDRCal0, dLDRCal, iLDRCal, nLDRCal)
    if (nTCalT > 0): PlotSubHistF11("TCalT", aTCalT, TCalT0, dTCalT, iTCalT, nTCalT)
    if (nTCalR > 0): PlotSubHistF11("TCalR", aTCalR, TCalR0, dTCalR, iTCalR, nTCalR)
    if (nNCal > 0): PlotSubHistF11("CalNoise", aNCal, 0, 1/nNCal, iNCal, nNCal)
    if (nNI > 0): PlotSubHistF11("SigNoise", aNI, 0, 1/nNI, iNI, nNI)


    plt.show()
    plt.close

    """
"""
    # only histogram
    #print("******************* " + aVar + " *******************")
    fig, ax = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(10, 10))
    iLDR = -1
    for LDRTrue in LDRrange:
        iLDR = iLDR + 1
        LDRmin[iLDR] = np.min(aLDRcorr[iLDR,:])
        LDRmax[iLDR] = np.max(aLDRcorr[iLDR,:])
        Rmin = np.min(aLDRcorr[iLDR,:])    * 0.999
        Rmax = np.max(aLDRcorr[iLDR,:])    * 1.001
        plt.subplot(5,2,iLDR+1)
        (n, bins, patches) = plt.hist(aLDRcorr[iLDR,:],
                 range=[Rmin, Rmax],
                 bins=200, log=False, alpha=0.2, density=False, color = '0.5', histtype='stepfilled')
        plt.tick_params(axis='both', labelsize=9)
        plt.plot([LDRTrue, LDRTrue], [0, np.max(n)], 'r-', lw=2)
    plt.show()
    plt.close
     # --- End of Plot F11 histograms
    """


"""
    # --- Plot K over LDRCal
    fig3 = plt.figure()
    plt.plot(LDRCal0+aLDRCal*dLDRCal/nLDRCal,aGHK[4,:], linewidth=2.0, color='b')

    plt.xlabel('LDRCal', fontsize=18)
    plt.ylabel('K', fontsize=14)
    plt.title(LID, fontsize=18)
    plt.show()
    plt.close
    """

# Additional plot routines ======>
"""
#******************************************************************************
# 1. Plot LDRCorrected - LDR(measured Icross/Iparallel)
LDRa = np.arange(1.,100.)*0.005
LDRCorra = np.arange(1.,100.)
if Y == - 1.: LDRa = 1./LDRa
LDRCorra = (1./Eta*LDRa*(GT+HT)-(GR+HR))/((GR-HR)-1./Eta*LDRa*(GT-HT))
if Y == - 1.: LDRa = 1./LDRa
#
#fig = plt.figure()
plt.plot(LDRa,LDRCorra-LDRa)
plt.plot([0.,0.5],[0.,0.5])
plt.suptitle('LDRCorrected - LDR(measured Icross/Iparallel)', fontsize=16)
plt.xlabel('LDR', fontsize=18)
plt.ylabel('LDRCorr - LDR', fontsize=16)
#plt.savefig('test.png')
#
"""
"""
#******************************************************************************
# 2. Plot LDRsim (simulated measurements without corrections = Icross/Iparallel) over LDRtrue
LDRa = np.arange(1.,100.)*0.005
LDRsima = np.arange(1.,100.)

atruea = (1.-LDRa)/(1+LDRa)
Ita = TiT*TiO*IinL*(GT+atruea*HT)
Ira = TiR*TiO*IinL*(GR+atruea*HR)
LDRsima = Ira/Ita  # simulated uncorrected LDR with Y from input file
if Y == -1.: LDRsima = 1./LDRsima
#
#fig = plt.figure()
plt.plot(LDRa,LDRsima)
plt.plot([0.,0.5],[0.,0.5])
plt.suptitle('LDRsim (simulated measurements without corrections = Icross/Iparallel) over LDRtrue', fontsize=10)
plt.xlabel('LDRtrue', fontsize=18)
plt.ylabel('LDRsim', fontsize=16)
#plt.savefig('test.png')
#
"""
