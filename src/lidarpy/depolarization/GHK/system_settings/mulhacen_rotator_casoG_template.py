# This Python script will be executed from within the main lidar_correction_ghk.py
# Probably it will be better in the future to let the main script rather read a conguration file,
# which might improve the portability of the code within an executable.
# Due to problems I had with some two letter variables, most variables are now with at least
# three letters mixed small and capital.

# Do you want to calculate the errors? If not, just the GHK-parameters are determined.
Error_Calc = False

# Header to identify the lidar system
#
EID = "gr"  # Earlinet station ID
LID = "mulhacen"  # Additional lidar ID (short descriptive text)
print("Lidar system :", EID, ", ", LID)

# +++ IL Laser and +-Uncertainty
bL = 1.0  # degree of linear polarization; default 1
Qin, dQin, nQin = (
    1.0,
    0.0,
    0,
)  # second Stokes vector parameter; default 1 => linear polarization  0.999 => LDR = 0.0005
Vin, dVin, nVin = (
    0.0,
    0.0,
    0,
)  # fourth Stokes vector parameter; default 0  => corresponds to LDR 0.0005 with DOP 1
RotL, dRotL, nRotL = (
    alpha_value,
    alpha_error,
    1,
)  # alpha; rotation of laser polarization in degrees; default 0

# +++ ME Emitter optics and +-Uncertainty;  default = no emitter optics
DiE, dDiE, nDiE = 0.0, 0.1, 1  # Diattenuation
TiE = 1.0  # Unpolarized transmittance
RetE, dRetE, nRetE = 0.0, 180.0, 1  # Retardance in degrees
RotE, dRotE, nRotE = 0.0, 1.0, 1  # beta: Rotation of optical element in degrees

# +++ MO Receiver optics including telescope
DiO, dDiO, nDiO = 0.88, 0.01, 1
TiO = 1.0
RetO, dRetO, nRetO = 0.0, 180.0, 1
RotO, dRotO, nRotO = 0.0, 0.5, 1  # gamma: Rotation of optical element in degrees

# +++++ PBS MT Transmitting path defined with TS, TP, PolFilter extinction ratio ERaT, and +-Uncertainty
#   --- Polarizing beam splitter transmitting path
TP, dTP, nTP = 0.95, 0.01, 1
TS, dTS, nTS = 0.005, 0.005, 1
RetT, dRetT, nRetT = 0.0, 180.0, 0  # Retardance in degrees
#   --- Pol.Filter behind transmitted path of PBS
ERaT, dERaT, nERaT = 1e-3, 5e-4, 1  # Extinction ratio
RotaT, dRotaT, nRotaT = (
    0.0,
    1.0,
    1,
)  # Rotation of the Pol.-filter in degrees; usually 0° because TP >> TS, but for PollyXTs it can also be 90°
#   --
TiT = 0.5 * (TP + TS)
DiT = (TP - TS) / (TP + TS)
DaT = (1 - ERaT) / (1 + ERaT)
TaT = 0.5 * (1 + ERaT)

# +++++ PBS MR Reflecting path defined with RS, RP, PolFilter extinction ratio ERaR  and +-Uncertainty
#   ---- for PBS without absorption the change of RS and RP must depend on the change of TP and TS. Hence the values and uncertainties are not independent.
RS_RP_depend_on_TS_TP = True
#   --- Polarizing beam splitter reflecting path
if RS_RP_depend_on_TS_TP:
    RP, dRP, nRP = 1 - TP, 0.00, 1  # do not change this
    RS, dRS, nRS = 1 - TS, 0.00, 1  # do not change this
else:
    RP, dRP, nRP = 0.05, 0.01, 0  # change this if RS_RP_depend_on_TS_TP = False
    RS, dRS, nRS = 0.98, 0.01, 0  # change this if RS_RP_depend_on_TS_TP = False
RetR, dRetR, nRetR = 0.0, 180.0, 0
#   --- Pol.Filter behind reflected path of PBS
ERaR, dERaR, nERaR = 1e-3, 5e-4, 1  # Extinction ratio
RotaR, dRotaR, nRotaR = (
    90.0,
    0.0,
    1,
)  # Rotation of the Pol.-filter in degrees; usually 90° because RS >> RP, but for PollyXTs it can also be 0°
#   --
TiR = 0.5 * (RP + RS)
DiR = (RP - RS) / (RP + RS)
DaR = (1 - ERaR) / (1 + ERaR)
TaR = 0.5 * (1 + ERaR)

# +++ Parallel signal detected in the transmitted channel => Y = +1, or in the reflected channel => Y = -1
Y = -1.0

# +++ Calibrator Location
LocC = 4  # location of calibrator: 1 = behind laser; 2 = behind emitter; 3 = before receiver; 4 = before PBS
# --- Calibrator Type used; defined by matrix values below
TypeC = 1  # Type of calibrator: 1 = mechanical rotator; 2 = hwp rotator (fixed retardation); 3 = linear polarizer; 4 = qwp; 5 = circular polarizer; 6 = real HWP calibration +-22.5°
# --- MC Calibrator parameters
if TypeC == 1:  # mechanical rotator
    DiC, dDiC, nDiC = 0.0, 0.0, 0
    TiC = 1.0
    RetC, dRetC, nRetC = 0.0, 0.0, 0
    RotC, dRotC, nRotC = (
        epsilon_value,
        epsilon_error,
        1,
    )  # constant calibrator offset epsilon
    # Rotation error without calibrator: if False, then epsilon = 0 for normal measurements
    RotationErrorEpsilonForNormalMeasurements = (
        True  # 	is in general True for TypeC == 1 calibrator
    )
elif TypeC == 2:  # HWP rotator
    DiC, dDiC, nDiC = 0.0, 0.0, 0
    TiC = 1.0
    RetC, dRetC, nRetC = 180.0, 0.0, 0
    # NOTE: use here twice the HWP-rotation-angle
    RotC, dRotC, nRotC = 0.0, 0.1, 1  # constant calibrator offset epsilon
    RotationErrorEpsilonForNormalMeasurements = (
        True  # 	is in general True for TypeC == 2 calibrator
    )
elif (
    TypeC == 3
):  # linear polarizer calibrator. Diattenuation DiC = (1-ERC)/(1+ERC); ERC = extinction ratio of calibrator
    DiC, dDiC, nDiC = 0.9998, 0.0001, 1  # ideal 1.0
    TiC = 0.4  # ideal 0.5
    RetC, dRetC, nRetC = 0.0, 0.0, 0
    RotC, dRotC, nRotC = 0.0, 0.1, 0  # constant calibrator offset epsilon
    RotationErrorEpsilonForNormalMeasurements = (
        False  # 	is in general False for TypeC == 3 calibrator
    )
elif TypeC == 4:  # QWP calibrator
    DiC, dDiC, nDiC = 0.0, 0.0, 0  # ideal 1.0
    TiC = 1.0  # ideal 0.5
    RetC, dRetC, nRetC = 90.0, 0.0, 0
    RotC, dRotC, nRotC = 0.0, 0.1, 1  # constant calibrator offset epsilon
    RotationErrorEpsilonForNormalMeasurements = (
        False  # 	is  False for TypeC == 4 calibrator
    )
elif (
    TypeC == 6
):  # real half-wave plate calibration at +-22.5°  => rotated_diattenuator_X22x5deg.odt
    DiC, dDiC, nDiC = 0.0, 0.0, 0
    TiC = 1.0
    RetC, dRetC, nRetC = 180.0, 0.0, 0
    # Note: use real HWP angles here
    RotC, dRotC, nRotC = 0.0, 0.1, 1  # constant calibrator offset epsilon
    RotationErrorEpsilonForNormalMeasurements = (
        True  # 	is in general True for TypeC == 6 calibrator
    )
else:
    print("calibrator not implemented yet")
    sys.exit()

# --- LDRCal assumed atmospheric linear depolarization ratio during the calibration measurements in calibration range with almost clean air (first guess)
LDRCal, dLDRCal, nLDRCal = 0.00355, 0.0005, 1  # spans the interference filter influence

# ====================================================
# NOTE: there is no need to change anything below.

# --- LDRtrue for simulation of measurement => LDRsim
LDRtrue = 0.4
LDRtrue2 = 0.004

# --- measured LDRm will be corrected with calculated parameters GHK
LDRmeas = 0.3

# --- this is just for correct transfer of the variables to the main file
RotL0, dRotL, nRotL = RotL, dRotL, nRotL
# Emitter
DiE0, dDiE, nDiE = DiE, dDiE, nDiE
RetE0, dRetE, nRetE = RetE, dRetE, nRetE
RotE0, dRotE, nRotE = RotE, dRotE, nRotE
# Receiver
DiO0, dDiO, nDiO = DiO, dDiO, nDiO
RetO0, dRetO, nRetO = RetO, dRetO, nRetO
RotO0, dRotO, nRotO = RotO, dRotO, nRotO
# Calibrator
DiC0, dDiC, nDiC = DiC, dDiC, nDiC
RetC0, dRetC, nRetC = RetC, dRetC, nRetC
RotC0, dRotC, nRotC = RotC, dRotC, nRotC
# PBS
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
