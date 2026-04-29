# -*- coding: utf-8 -*-
"""
- ENGINEERED BY: jararias y vlara
- LAST UPDATE: 20160602 [20160526, 20160426]
- VERSION: 12
- COMMENTS:
   +0.-This library is a compemdium of utilities to process solar radiation time series and data.
   +1.-Based mainly on Iqbal, 1983, but also with models of Rigollier, Ruiz-Arias, Perez, Blanco-Muriel
       Michalsky and MESOR.
   +WARNING: prestar note that parameter ls is allways for UTC => |0Âº|, then longitude > 0 <=> east
- INDEX OF DEFINITIONS:
   0.- class SUN: sun position parameters. Additional methods: get_ma(), get_kt()
   1.- class svSolis: Simplified Version of SOLIS clear-sky model. Main attributes: ghi, dni, dif
   2.- class REST2: REST2 clear-sky model. Main attributes: ghi, dni, dif
   3.- class JARARIAS: Ruiz-Arias diffuse fraction model. Main attributes: kd.
   4.- class DIRINDEX: SWS method for GHI to DNI calculation. Main attributes: dni, ghi (provided)
   5.- class BIRD: Bird and Hulstrom (1981) clear-sky model. Main attributes: ghi, dni, dif
   6.- class MesorBIRD: particularization of Brid model for clear and dry atmosphere
   7.-
   8.- clas projIrrad: Irradiance projected components over tracking surfaces.
   9.- class powerPV: PV energy
   10.- class ESRA
- USAGE:
   > python solar.py
   or
   > import python swsolib
   # To create SUN object
   > dates     = pl.drange(datetime(2000,6,21,0,0),datetime(2000,6,22,0,0),timedelta(minutes=1))
   > sunobject = swsolib.SUN(dates,-3.5,37.5,elev=600.0)
   # To create clear-sky object: whether svSolis or REST2
   > solis     = swsolib.svSolis(sunobject,w=water_column,p=1010.0,aod700=0.2)
   > rest2     = REST2(sunobject)
   # To create an object of JARARIAS class.
   > jararias  = JARARIAS(ghi,sunobject)
   # To create an object of DIRINDEX
   > DIRINDEX     = DIRINDEX(sat_ghi,sunobject,solis,method='dirint')
   # To create an object of BIRD
   > bird  = BIRD(sunobject)

NOTES:

Add:
    log('generando posiciones del dia ...')
    horizon = np.where(np.diff(np.sign(sol.get_csza())))[0]
    midday = np.diff(np.sign(np.diff(sol.get_csza())))
    midday = np.where(midday>=0,0,1)
    midday = np.where(midday)[0]
    log(' OK\n')
"""
# -------------------------------------------------------------------------------------------
import numpy as np
import pylab as pl

# import warnings
from matplotlib.dates import num2date, date2num

# -------------------------------------------------------------------------------------------


def _checkinput(value, variable, limits, reference, dtype):
    _value = np.array(value, dtype=dtype)
    _reference = np.array(reference, dtype=dtype)
    if _value.ndim == 0:
        _value = value * np.ones(_reference.shape)
    elif _value.shape != _reference.shape:
        raise ValueError(
            "expected shape %s in %s. Got %s"
            % (repr(_reference.shape), variable, repr(_value.shape))
        )
    min_value, max_value = limits
    if min_value is not None:
        if np.any(_value < min_value):
            # warnings.warn('expected values greater than %f in %s. Smaller values have been \"clipped\"'%(min_value,variable))
            _value = np.where(_value < min_value, min_value, value)
    if max_value is not None:
        if np.any(_value > max_value):
            # warnings.warn('expected values smaller than %f in %s. Greater values have been \"clipped\"'%(max_value,variable))
            _value = np.where(_value > max_value, max_value, value)
    return _value


class SUN(object):
    """
    Paper de Blanco-Muriel (Plataforma Solar de Almeria).
    Adaptado del codigo fortran de Chris Gueymard, Dec 2011.
    GUARNING: el criterio de signos y los valores de los ángulos no coinciden con Iqbal. Cosas a considerar:
        Declinacion:
            declinacion ~= declinacion Iqbal
        Acimut: criterio de signos de Iqbal es Sur = 0, Este +, Oeste -
            sin(180-get_saa()) = sin(Acimut Iqbal)
            cos(180-get_saa()) = cos(Acimut Iqbal)
            SAA: sentido horario desde el norte (0), este (90), sur (180),
            oeste (270)
        Angulo horario: criterio de signos de Iqbal es medio dia solar = 0, por la mañana +
            sin(-1*sol.get_hour_angle()) = sin(w) Iqbal
            cos(-1*sol.get_hour_angle()) = cos(w) Iqbal
    GUARNING: cuidado con los valores horarios. Una aproximacion adecuada para el calculo de la posicion solar
    promedio es usar el times en la mitad del intervalo horario.
    """

    def __init__(
        self, times, longitude, latitude, elev=0.0, is_tst=False, std_longitude=0.0
    ):
        # times: [datetime o float] puede ser TST o UTC, y eso se indica con el argumento is_tst
        #        El algoritmo lo transformara siempre a UTC, ya que trabaja en ese sistema de referencia.
        #        Asi, el atributo .dates devuelve las fechas (numericas) metidas como times (es una copia),
        #        mientras que el metodo get_utc devuelve las fechas (numericas) en UTC con las que trabaja
        #        el algoritmo.
        #        Si se mete un times con is_tst=True es mejor recuperarlo llamando al atributo .dates ya que
        #        el metodo get_tst siempre hace una operacion que mete un desajuste (de centesimas de segundo)
        #        pero que hace que un dato introducido como las 10:00 TST se devuelva con get_tst como
        #        09:59:59.970445+00:00. Como el atributo .dates es una copia, devolvera (numerica) la fecha.
        # std_longitude: el meridiano estandar mas cercano. Para Spain, el 0.
        self._lat = None
        self._lon = None
        self._elev = elev
        self._dtnum = None
        self._year = None
        self._month = None
        self._day = None
        self._hour = None
        self._djul = None
        self._decli = None
        self._hourangl = None
        self._righta = None
        self._sunlng = None

        self._lat = np.asarray(latitude, dtype=np.float64)
        self._lon = np.asarray(longitude, dtype=np.float64)
        if self._lat.shape != self._lon.shape:
            raise ValueError("latitude and longitude must have same shape")

        times = np.asarray(times)
        if isinstance(times[0], float):
            pass
        else:
            times = date2num(times)
        self.dates = np.copy(times)  # date-time in "times" reference (UTC or TST)
        if is_tst == True:
            # TST = UTC + DL + EOT  =>  UTC = TST - DL - EOT
            # donde:
            #  DL  = (Long - Long_std) * 4, [minutos]
            #  EOT = Equation Of Time
            DL = 4.0 * (self._lon - std_longitude) / (24.0 * 60.0)
            times = times - DL
            self._update_times(times)  # actualizamos para calcular la EOT
            self._celestial_and_local_coordinates()
            EOT = self.get_eot() / (24.0 * 60.0)
            times = times - EOT
            self._update_times(times)
            self._celestial_and_local_coordinates()
        else:
            self._update_times(times)
            self._celestial_and_local_coordinates()

    def _update_times(self, times):
        self._dtnum = times
        dtimes = pl.num2date(times)
        self._year = np.asarray([t.year for t in dtimes], dtype=np.float64)
        self._month = np.asarray([t.month for t in dtimes], dtype=np.float64)
        self._day = np.asarray([t.day for t in dtimes], dtype=np.float64)
        self._hour = np.asarray(24.0 * (times - np.floor(times)), dtype=np.float64)
        self._djul = np.asarray(
            [t.timetuple().tm_yday for t in dtimes], dtype=np.float64
        )
        if self._lat.ndim == 2:
            self._year = self._year[:, np.newaxis, np.newaxis]
            self._month = self._month[:, np.newaxis, np.newaxis]
            self._day = self._day[:, np.newaxis, np.newaxis]
            self._hour = self._hour[:, np.newaxis, np.newaxis]
            self._djul = self._djul[:, np.newaxis, np.newaxis]

    def _celestial_and_local_coordinates(self):
        # julian day
        liaux1 = (self._month - 14.0) / 12.0
        liaux2 = (
            (1461.0 * (self._year + 4800.0 + liaux1)) / 4.0
            + (367.0 * (self._month - 2.0 - 12 * liaux1)) / 12.0
            - (3.0 * ((self._year + 4900.0 + liaux1) / 100.0)) / 4.0
            + self._day
            - 32075.0
        )
        julian = liaux2 - 0.5 + self._hour / 24.0
        # ecliptic coordinates
        elapsd = julian - 2451545.0
        omega = 2.1429 - 0.0010394594 * elapsd
        sunlng = 4.8950630 + 0.017202791698 * elapsd
        anomly = 6.2400600 + 0.0172019699 * elapsd
        eclipl = (
            sunlng
            + 0.03341607 * np.sin(anomly)
            + 0.00034894 * np.sin(2.0 * anomly)
            - 0.0001134
            - 0.0000203 * np.sin(omega)
        )
        eclip0 = 0.4090928 - 6.2140e-9 * elapsd + 0.0000396 * np.cos(omega)
        # celestial coordinates
        sinelg = np.sin(eclipl)
        dy = np.cos(eclip0) * sinelg
        dx = np.cos(eclipl)
        righta = np.arctan2(dy, dx)
        righta = np.where(righta < 0.0, righta + 2 * np.pi, righta)
        self._decli = np.arcsin(np.sin(eclip0) * sinelg)
        # local coordinates
        gmst = 6.6974243242 + 0.0657098283 * elapsd + self._hour
        lmst = np.deg2rad(gmst * 15.0 + self._lon)
        self._gmst = gmst
        self._lmst = lmst
        self._hourangl = lmst - righta
        self._righta = righta
        self._sunlng = sunlng

    def get_eccentricity(self):
        da = 2.0 * np.pi * (self._djul - 1.0) / 365.0
        ec = (
            1.000110
            + 0.034221 * np.cos(da)
            + 0.001280 * np.sin(da)
            + 0.000719 * np.cos(2 * da)
            + 0.000077 * np.sin(2 * da)
        )
        if ec.size == 1:
            return np.asscalar(ec)
        return ec

    def get_hour_angle(self):
        """
        In degrees
        """
        if self._hourangl.size == 1:
            return np.asscalar(np.rad2deg(self._hourangl))
        return np.rad2deg(self._hourangl)

    def get_declination(self):
        """
        In degrees
        """
        if self._decli.size == 1:
            return np.asscalar(np.rad2deg(self._decli))
        return np.rad2deg(self._decli)

    def get_sza(self):
        radius = 6371.01
        aunit = 149597890.0
        lat = np.deg2rad(self._lat)
        z = np.arccos(
            np.sin(lat) * np.sin(self._decli)
            + np.cos(lat) * np.cos(self._decli) * np.cos(self._hourangl)
        )
        # ~ # parallax correction
        paralax = (radius / aunit) * np.sin(z)
        z = z + paralax
        if z.size == 1:
            return np.asscalar(np.rad2deg(z))
        return np.rad2deg(z)

    def get_sza_at_noon(self):
        radius = 6371.01
        aunit = 149597890.0
        lat = np.deg2rad(self._lat)
        z = np.arccos(
            np.sin(lat) * np.sin(self._decli) + np.cos(lat) * np.cos(self._decli)
        )
        # parallax correction
        paralax = (radius / aunit) * np.sin(z)
        z = z + paralax
        if z.size == 1:
            return np.asscalar(np.rad2deg(z))
        return np.rad2deg(z)

    def get_csza(self):
        z = np.deg2rad(self.get_sza())
        if z.size == 1:
            return np.asscalar(np.cos(z))
        return np.cos(z)

    def get_saa(self):
        lat = np.deg2rad(self._lat)
        dy = -np.sin(self._hourangl)
        dx = np.tan(self._decli) * np.cos(lat) - np.sin(lat) * np.cos(self._hourangl)
        azimu = np.arctan2(dy, dx)
        azimu = np.where(azimu < 0.0, azimu + 2.0 * np.pi, azimu)
        if azimu.size == 1:
            return np.asscalar(np.rad2deg(azimu))
        return np.rad2deg(azimu)

    def get_eot(self):
        # from Michalsky
        righta = np.rad2deg(self._righta)
        sunlng = np.rad2deg(self._sunlng)
        xsun = -np.floor(np.abs(sunlng) / 360.0)
        xsun = np.where(sunlng < 0.0, -xsun + 1, xsun)
        eot = (sunlng + xsun * 360.0 - righta) * 4.0
        eot = np.where(eot > 20.0, eot - 1440.0, eot)
        eot = np.where(eot < -20.0, eot + 1440.0, eot)
        if eot.size == 1:
            return float(np.asscalar(eot))
        return eot.astype(np.float)

    def get_tst(self):
        return self._dtnum + 4.0 * self._lon / 1440.0 + self.get_eot() / 1440.0

    def get_utc(self):
        return self._dtnum

    def get_local_time(self):
        return self.get_tst()

    def get_sza_refrac(self, pressure, temperature):
        pres = np.asarray(pressure)
        temp = np.asarray(temperature)
        eld = 90.0 - self.get_sza()
        eld2 = eld**2
        pt = pres / temp
        refrac = np.where(
            (eld < 15.0) & (eld >= -2.5),
            pt
            * (0.1594 + 0.0196 * eld + 2e-5 * eld2)
            / (1.0 + 0.505 * eld + 0.0845 * eld2),
            0.00452 * pt / np.tan(np.deg2rad(eld)),
        )
        z = 90.0 - (eld + refrac)
        if z.size == 1:
            return np.asscalar(z)
        return z

    def get_ma(self, model="kasten"):
        """
        Returns the relative air mass.
        > model [string]: it must be
            + esra (Rigollier et al., 2000 model (ESRA))
            + bird (Bird model (model C in Iqbal 1989))
            + kasten (Kasten and Young 1989 model) (DEFAULT)
        """
        _limit = {"air_mass": (1.0e-6, 40.0), "cosZ": (1.0e-6, 1.0)}
        _dtype = np.float64
        cosZ = np.clip(
            np.array(self.get_csza(), dtype=_dtype),
            _limit["cosZ"][0],
            _limit["cosZ"][1],
        )
        if model == "bird":
            m = np.exp(-self._elev / 8446.0) / (
                cosZ + 0.15 * ((93.885 - np.rad2deg(np.arccos(cosZ))) ** (-1.253))
            )
        elif model == "kasten":
            m = (np.exp(self._elev / -8434.5)) / (
                cosZ + 0.50572 * (96.07995 - np.rad2deg(np.arccos(cosZ))) ** (-1.6364)
            )
        elif model == "esra":
            a = np.arcsin(cosZ)
            aRefr = (np.rad2deg(0.061359)) * (
                (0.1594 + (1.1230 * a) + (0.065656 * (a**2)))
                / (1.0 + (28.9344 * a) + 277.3971 * (a**2))
            )
            at = (np.rad2deg(a)) + aRefr
            m = (np.exp(self._elev / (-8434.5))) / (
                np.sin(np.radians(at)) + (0.50572 * ((at + 6.07995) ** (-1.6364)))
            )
        else:
            raise ValueError(
                "\n\n!0_/\tmacagoen-ERROR <swsolib.SUN>: model %s is neither: esra, bird nor kasten.\n\n"
                % model
            )
        return np.clip(
            np.array(m, dtype=_dtype), _limit["air_mass"][0], _limit["air_mass"][1]
        )

    def get_kt(self, rad, model="kasten"):
        """
        It returns the the angle independet clearness index (kt) proposed by Perez et al. 1990.
        > rad [sequence]: Float surface irradiance.
        """
        # _limit={'rad': (None,None)}
        # _dtype=np.float64
        _limit = {
            "kt": (0.0, 1.0),
            "air_mass": (1.0e-6, 40.0),
            "cosZ": (1.0e-6, 1.0),
            "eccentricity": (0.96, 1.04),
            "rad": (0.0, None),
        }
        _dtype = np.float64
        ecc = np.clip(
            np.array(self.get_eccentricity(), dtype=_dtype),
            _limit["eccentricity"][0],
            _limit["eccentricity"][1],
        )
        cosZ = np.clip(
            np.array(self.get_csza(), dtype=_dtype),
            _limit["cosZ"][0],
            _limit["cosZ"][1],
        )
        if (model != "esra") and (model != "kasten") and (model != "bird"):
            raise ValueError(
                "\n\n!0_/\tmacagoen-ERROR <swsolib.SUN>: model %s is neither: esra, bird nor kasten.\n\n"
                % model
            )
        rad0 = 1366.1 * ecc * cosZ
        rad = _checkinput(rad, "rad", _limit["rad"], rad0, dtype=_dtype)
        kt = rad / rad0
        ma = self.get_ma(model)
        return np.clip(
            np.array(
                (kt / ((1.031 * np.exp(-1.4 / (0.9 + (9.4 / ma)))) + 0.1)), dtype=_dtype
            ),
            _limit["kt"][0],
            _limit["kt"][1],
        )

    def get_toa(self):
        """
        Top Of Atmosphere Irradiation

        :return:
        """
        _limit = {
            "air_mass": (1.0e-6, 40.0),
            "cosZ": (1.0e-6, 1.0),
            "eccentricity": (0.96, 1.04),
            "rad": (0.0, None),
        }
        _dtype = np.float64
        ecc = np.clip(
            np.array(self.get_eccentricity(), dtype=_dtype),
            _limit["eccentricity"][0],
            _limit["eccentricity"][1],
        )
        cosZ = np.clip(
            np.array(self.get_csza(), dtype=_dtype),
            _limit["cosZ"][0],
            _limit["cosZ"][1],
        )
        return 1366.1 * ecc * cosZ


class svSOLIS(object):
    """
    Ineichen, P., 2008: A broadband simplified version of the Solis clear sky model, Solar Energy, 82, pp. 758-762
    """

    def __init__(self, sun, w=1.4, p=1013.25, aod700=0.1):
        """
               It returns an object whose attributes are the ghi, dni and dif values (also dif_orig).
               > sun [object]: SUN object (position of the Sun).
               > w [single or sequence]: values of the water vapor column content (in cm). If a single value is provided then it is
                                         assumed for all dates time series. If sequence it must have the same lenght than dates.
                                         If it is not provided (default) a standard value of 1.0 is used.
                                         It is limited to the interval [0.2, 10].
               aod700 : aod a 700 nm
               p      : presion atmosferica [hPa] or [mbar]
               Assumed: O3 = 340 Dobson units,
                        urban? aerosol type, -que raro que haya basado el modelo en un tipo urbano!!
                        elevation range 0-7000 meters






        It returns an SVSolisBox object whose attributes are the ghi, dni and dif values.
          > dates [sequence]: date-times time series in float or datetime format.
          > alpha [sequence]: solar elevation angle (in radians) time series, corresponding to dates.
                              It is denoted as h in the original paper (Ineichen, 2008).
          > H2O [single or sequence]: values of the water vapor column content [cm]. If a single value is provided then it is
                                      assumed for all dates time series. If sequence it must have the same lenght than dates.
                                      If it is not provided (default) a standard value of 1.0 is used.
                                      It is limited to the interval [0.2, 10].
          > AOD700 [single or sequence]: values of the aerosol optical depth. If a single value is provided then it is
                                      assumed for all dates time series. If sequence it must have the same lenght than dates.
                                      If it is not provided (default) a standard value of 0.1 is used.
                                      It is limited to the interval [0, 0.45].
          > P [single or sequence]: values of the atmospheric pressure at considered altitude (in the same units than Po).
                                    If a single value is provided then it is assumed for all dates time series. If sequence
                                    it must have the same lenght than dates.
                                    If it is not provided (default) a standard value of 1000 hPa is used.
          > Po [single]           : values of the atmospheric pressure at sea level (in the same units than P).
                                    The provided value is assumed as a constant value.
                                    If it is not provided (default) a standard value of 1013.25 hPa is used.
          GUARNING: use the same units for P and Po
          NOTE: unlike the paper, diffuse componet is also derived in analytical way from GHI and DNI. This is because an strange behaviour
                is detected at sunrise and sunset time for the diffuse component which appears over the GHI, when the paper formula is used.
                Then two value for the diffuse component are provided: .diff_orig and .dif; the last one is the derived.
          NOTE: remember that solar radiation is computed at the same time reference as alpha. dates is only used to
                obtain the eccentricity correction factor.
          NOTE: the model is obtained for:
                   + O3 = 340 Dobson units
                   + aerosol type: urban
                   + elevation: [0, 7000] meters


        """
        _limit = {
            "precipitable water": (0.2, 10.0),  # cm
            "aod@700": (0.0, 0.45),
            "pressure": (850.0, 1020.0),
        }  # hPa, limites que he puesto yo
        _dtype = np.float64

        po = 1013.25  # hPa
        self.cosz = np.array(sun.get_csza(), dtype=_dtype)
        ecc = np.array(sun.get_eccentricity(), dtype=_dtype)
        w = _checkinput(
            w,
            "precipitable water",
            _limit["precipitable water"],
            self.cosz,
            dtype=_dtype,
        )
        aod = _checkinput(aod700, "aod@700", _limit["aod@700"], self.cosz, dtype=_dtype)
        p = _checkinput(p, "pressure", _limit["pressure"], self.cosz, dtype=_dtype)

        lnppo = np.log(p / po)
        lnw = np.log(w)

        is_day = self.cosz > 1e-6

        Io = 1366.1 * ecc
        Io0 = 1.08 * (w**0.0051)
        Io1 = 0.97 * (w**0.032)
        Io2 = 0.12 * (w**0.56)
        Iop = Io * (Io2 * (aod**2) + Io1 * aod + Io0 + 0.071 * (lnppo))

        lnw = lnw
        tg1 = 1.24 + 0.047 * lnw + 0.0061 * (lnw**2)
        tg0 = 0.27 + 0.043 * lnw + 0.0090 * (lnw**2)
        tgp = 0.0079 * w + 0.1
        tg = tg1 * aod + tg0 + tgp * (lnppo)
        g = -0.0147 * lnw - 0.3079 * (aod**2) + 0.2846 * aod + 0.3798
        self.ghi = np.zeros(self.cosz.shape)
        self.ghi[is_day] = (
            Iop[is_day]
            * np.exp(-tg[is_day] / (self.cosz[is_day] ** g[is_day]))
            * self.cosz[is_day]
        )

        tb1 = 1.82 + 0.056 * lnw + 0.0071 * (lnw**2)
        tb0 = 0.33 + 0.045 * lnw + 0.0096 * (lnw**2)
        tbp = 0.0089 * w + 0.13
        tb = tb1 * aod + tb0 + tbp * (lnppo)
        b1 = 0.00925 * (aod**2) + 0.0148 * aod - 0.0172
        b0 = -0.7565 * (aod**2) + 0.5057 * aod + 0.4557
        b = b1 * lnw + b0
        self.dni = np.zeros(self.cosz.shape)
        self.dni[is_day] = Iop[is_day] * np.exp(
            -tb[is_day] / (self.cosz[is_day] ** b[is_day])
        )

        # la difusa para angulos cenitales muy elevados sale mayor que la global!! No tiene sentido
        # asi que la calculo como ghi-dhi*cosz
        self.dif = self.ghi - self.dni * self.cosz

        td4 = np.where(aod < 0.05, 86.0 * w - 13800.0, -0.21 * w + 11.6)
        td3 = np.where(aod < 0.05, -3.11 * w + 79.4, 0.27 * w - 20.7)
        td2 = np.where(aod < 0.05, -0.23 * w + 74.8, -0.134 * w + 15.5)
        td1 = np.where(aod < 0.05, 0.092 * w - 8.86, 0.0554 * w - 5.71)
        td0 = np.where(aod < 0.05, 0.0042 * w + 3.12, 0.0057 * w + 2.94)
        tdp = np.where(
            aod < 0.05,
            -0.83 * ((1.0 + aod) ** (-17.2)),
            -0.71 * ((1.0 + aod) ** (-15.0)),
        )
        td = (
            td4 * (aod**4)
            + td3 * (aod**3)
            + td2 * (aod**2)
            + td1 * aod
            + td0
            + tdp * (lnppo)
        )
        dp = 1.0 / (18.0 + 152.0 * aod)
        d = -0.337 * (aod**2) + 0.63 * aod + 0.116 + dp * (lnppo)
        self.dif_orig = np.zeros(self.cosz.shape)
        self.dif_orig[is_day] = Iop[is_day] * np.exp(
            -td[is_day] / (self.cosz[is_day] ** d[is_day])
        )


class REST2(object):
    """
    Gueymard, C.A., 2008: REST2: High-performance solar radiation model for cloudless-sky irradiance, illuminance,
        and photosynthetically active radiation -Validation with a benchmark dataset, Solar Energy, 82, pp. 272-285.

    Coefficients updates from REST2 code version 8.4 (Dec, 2011)
    TODO: double-check coefficients in newest REST2 version
          Ebn2 starts at a positive value at sunrise, is that a bug? Double-check with REST2 executions
          PAR?
          Illuminances?
    """

    def __init__(
        self,
        sun,
        w=1.4,
        p=1013.25,
        beta=0.06,
        alpha1=1.3,
        alpha2=1.3,
        uo=0.35,
        un=0.0002,
        albedo=0.2,
    ):
        """
        sun    : SUN object (position of the Sun).
        w      : agua precipitable [cm]
        p      : presion atmosferica [hPa]
        beta   : Angstrom turbidity. Aerosol optical depth at 1 micron
        alpha1 : Angstrom turbidity exponent for band 1 (wvl < 700 nm)
        alpha2 : Angstrom turbidity exponent for band 2 (wvl > 700 nm)
        uo     : ozone amount [atm-cm]
        un     : nitrogen dioxide amount [atm-cm]
        albedo : ground albedo
        Assumed: O3 = 340 Dobson units,
                 urban? aerosol type, -que raro que haya basado el modelo en un tipo urbano!!
                 elevation range 0-7000 meters
        """
        _dtype = np.float128
        _limit = {
            "precipitable water": (0.0, 10.0),  # cm
            "alpha1": (0.0, 2.5),
            "alpha2": (0.0, 2.5),
            "beta": (0.0, 1.1),
            "ozone": (0.0, 0.6),  # atm-cm
            "no2": (0.0, 0.3),  # atm-cm
            "albedo": (0.0, 1.0),
            "pressure": (300.0, 1100.0),
        }  # hPa
        po = 1013.25  # hPa
        Eon = 1366.1  # W/m2
        F1 = 0.46512
        F2 = 0.51951

        self.cosz = np.array(sun.get_csza(), dtype=_dtype)
        # self.cosz = cosz
        ecc = np.array(sun.get_eccentricity(), dtype=_dtype)
        w = _checkinput(
            w,
            "precipitable water",
            _limit["precipitable water"],
            self.cosz,
            dtype=_dtype,
        )
        p = _checkinput(p, "pressure", _limit["pressure"], self.cosz, dtype=_dtype)
        oz = _checkinput(uo, "ozone", _limit["ozone"], self.cosz, dtype=_dtype)
        no2 = _checkinput(un, "no2", _limit["no2"], self.cosz, dtype=_dtype)
        alpha1 = _checkinput(
            alpha1, "alpha1", _limit["alpha1"], self.cosz, dtype=_dtype
        )
        alpha2 = _checkinput(
            alpha2, "alpha2", _limit["alpha2"], self.cosz, dtype=_dtype
        )
        beta = _checkinput(beta, "beta", _limit["beta"], self.cosz, dtype=_dtype)
        albedo = _checkinput(
            albedo, "albedo", _limit["albedo"], self.cosz, dtype=_dtype
        )

        is_day = self.cosz > 0.0

        Eon = Eon * ecc
        omega1 = 0.95  # valor en el codigo fortran
        omega2 = 0.90  # valor en el codigo fortran
        beta1 = beta * (0.7 ** (alpha1 - alpha2))
        beta2 = beta
        albedo1 = albedo
        albedo2 = albedo

        air_mass = lambda p, cosz, z: np.maximum(
            1.0, 1.0 / (cosz + (p[0] * (z ** p[1])) / ((p[2] - z) ** p[3]))
        )
        sza = np.degrees(np.arccos(self.cosz, dtype=_dtype))
        mR = air_mass([0.48353, 0.095846, 96.741, 1.7540], self.cosz, sza)
        mo = air_mass([1.06510, 0.637900, 101.800, 2.2694], self.cosz, sza)
        mw = air_mass([0.10648, 0.114230, 93.781, 1.9203], self.cosz, sza)
        ma = air_mass([0.16851, 0.181980, 95.318, 1.9542], self.cosz, sza)
        mn = mw
        mRp = mR * p / po
        mp = 1.66

        # Transmittances for Band 1
        rational = lambda p, x: (p[0] + p[1] * x + p[2] * (x**2)) / (
            p[3] + p[4] * x + p[5] * (x**2)
        )
        transmittance = lambda p, x: np.minimum(1.0, np.maximum(0.0, rational(p, x)))
        TR1 = transmittance([1.0, 1.81690, -0.033454, 1.0, 2.06300, 0.319780], mRp)
        Tg1 = transmittance([1.0, 0.95885, 0.012871, 1.0, 0.96321, 0.015455], mRp)
        f1 = rational([10.979000, -8.542100, 0.0, 1.0, 2.0115, 40.189], oz) * oz
        f2 = rational([-0.027589, -0.005138, 0.0, 1.0, -2.4857, 13.942], oz) * oz
        f3 = rational([10.995000, -5.500100, 0.0, 1.0, 1.6784, 42.406], oz) * oz
        To1 = transmittance([1.0, f1, f2, 1.0, f3, 0.0], mo)
        g1 = rational([0.17499, 41.654, -2146.4, 1.0, 0.0, 22295.0], no2)
        g2 = rational([-1.21340, 59.324, 0.0, 1.0, 0.0, 8847.8], no2) * no2
        g3 = rational([0.17499, 61.658, 9196.4, 1.0, 0.0, 74109.0], no2)
        Tn1 = transmittance([1.0, g1, g2, 1.0, g3, 0.0], mn)
        Tnp1 = transmittance([1.0, g1, g2, 1.0, g3, 0.0], mp)
        h1 = rational([0.065445, 0.00029901, 0.0, 1.0, 1.2728, 0.0], w) * w
        h2 = rational([0.065687, 0.00132180, 0.0, 1.0, 1.2008, 0.0], w) * w
        Tw1 = transmittance([1.0, h1, 0.0, 1.0, h2, 0.0], mw)
        Twp1 = transmittance([1.0, h1, 0.0, 1.0, h2, 0.0], mp)

        # Transmittances for Band 2
        TR2 = transmittance([1.0, -0.010394, 0.0, 1.0, 0.0, -0.00011042], mRp)
        Tg2 = transmittance([1.0, 0.272840, -0.00063699, 1.0, 0.30306, 0.0], mRp)
        To2 = 1.0
        Tn2 = 1.0
        Tnp2 = 1.0
        c1 = rational([19.566, -1.6506, 1.0672, 1.0, 5.4248, 1.6005], w) * w
        c2 = rational([0.50158, -0.14732, 0.047584, 1.0, 1.1811, 1.0699], w) * w
        c3 = rational([21.286, -0.39232, 1.2692, 1.0, 4.8318, 1.412], w) * w
        c4 = rational([0.70992, -0.23155, 0.096514, 1.0, 0.44907, 0.75425], w) * w
        Tw2 = transmittance([1.0, c1, c2, 1.0, c3, c4], mw)
        Twp2 = transmittance([1.0, c1, c2, 1.0, c3, c4], mp)

        # Turbidity functions and equivalent aerosol wavelengths
        alpha13 = np.abs(alpha1 - 1.3) <= 1e-3
        d0 = np.where(
            alpha13,
            0.544474,
            rational([0.576640, -0.024743, 0.0, 1.0, 0.0, 0.0], alpha1),
        )
        d1 = np.where(
            alpha13,
            0.00877874,
            rational([0.093942, -0.226900, 0.12848, 1.0, 0.6418, 0.0], alpha1),
        )
        d2 = np.where(
            alpha13,
            0.196771,
            rational([-0.093819, 0.366680, -0.12775, 1.0, -0.11651, 0.0], alpha1),
        )
        d3 = np.where(
            alpha13,
            0.294559,
            rational([0.152320, -0.087214, 0.012664, 1.0, -0.90454, 0.26167], alpha1)
            * alpha1,
        )
        ua = np.log(1.0 + ma * beta1)  # page 274, below Eq (7a)
        wvle1 = np.clip(rational([d0, d1, d2, 1.0, 0.0, d3], ua), 0.3, 0.65)

        alpha13 = np.abs(alpha2 - 1.3) <= 1e-3
        e0 = np.where(
            alpha13,
            1.0380760,
            rational([1.18300, -0.022989, 0.0208290, 1.0, 0.11133, 0.0], alpha2),
        )
        e1 = np.where(
            alpha13,
            -0.1055590,
            rational([-0.50003, -0.183290, 0.2383500, 1.0, 1.67560, 0.0], alpha2),
        )
        e2 = np.where(
            alpha13,
            0.0643067,
            rational([-0.50001, 1.141400, 0.0083589, 1.0, 11.16800, 0.0], alpha2),
        )
        e3 = np.where(
            alpha13,
            -0.1092430,
            rational([-0.70003, -0.735870, 0.5150900, 1.0, 4.76650, 0.0], alpha2),
        )
        ua = np.log(1.0 + ma * beta2)  # page 274, below Eq (7a)
        wvle2 = np.clip(rational([e0, e1, e2, 1.0, 0.0, e3], ua), 0.75, 1.75)

        # Aerosol directional factor Ba, Eq (11)
        Ba = 1.0 - np.exp(-0.6931 - 1.8326 * self.cosz)

        # Rayleigh multiple scattering correction, Eq (10)
        BR1 = 0.4625 * (1.0024 - 0.0055808 * mR + 0.000051487 * mR * mR)
        BR2 = 0.588

        # Aerosol transmittances and albedos for the two bands
        taua1 = beta1 * np.power(wvle1, -alpha1)  # Eq (6)
        Ta1 = np.clip(np.exp(-ma * taua1), 1e-5, 1.0)  # Eq (7a)
        Tas1 = np.exp(-ma * taua1 * omega1)  # Eq (7b)
        taua2 = beta2 * np.power(wvle2, -alpha2)  # Eq (6)
        Ta2 = np.clip(np.exp(-ma * taua2), 1e-5, 1.0)  # Eq (7a)
        Tas2 = np.exp(-ma * taua2 * omega2)  # Eq (7b)

        # Band 1: visible (0.29 - 0.7 um)
        E0n1 = Eon * F1
        Ebn1 = TR1 * Tg1 * To1 * Tn1 * Tw1 * Ta1 * E0n1  # Eq (3)
        Edif11 = np.maximum(
            0.2,
            To1
            * Tg1
            * Tnp1
            * Twp1
            * BR1
            * (1.0 - TR1)
            * (Ta1**0.55)
            * E0n1
            * self.cosz,
        )
        betama = ma * beta2
        ratio = np.where(
            betama >= 0.199,
            4.7488 - 0.84236 * betama,
            4.58033 + np.exp((0.606 - 2.9998 * betama**0.57) / (0.2 - betama)),
        )
        Edif12 = np.maximum(
            0.2,
            Ba
            * To1
            * Tg1
            * Tnp1
            * Twp1
            * (1.0 - (Tas1**0.15))
            * TR1
            * ratio
            * E0n1
            * self.cosz,
        )
        Edif1 = Edif11 + Edif12
        Edir1 = Ebn1 * self.cosz

        # Band 2: infrared (0.7 - 4 um)
        E0n2 = Eon * F2
        Ebn2 = TR2 * Tg2 * To2 * Tn2 * Tw2 * Ta2 * E0n2  # Eq (3)
        Edif21 = np.maximum(
            0.1,
            To2
            * Tg2
            * Tnp2
            * Twp2
            * BR2
            * (1.0 - TR2)
            * (Ta2**0.55)
            * E0n2
            * self.cosz,
        )
        Edif22 = np.maximum(
            0.0,
            Ba
            * To2
            * Tg2
            * Tnp2
            * Twp2
            * (1.0 - (Tas2**0.15))
            * TR2
            * ratio
            * E0n2
            * self.cosz,
        )
        Edif2 = Edif21 + Edif22
        Edir2 = Ebn2 * self.cosz

        # Sky albedo for backscattered radiation, Band 1, Appendix 1
        x0 = rational([0.37567, 0.22946, 0.0, 1.0, -0.10832, 0.0], alpha1)
        x1 = rational([0.84057, 0.68683, 0.0, 1.0, -0.08158, 0.0], alpha1)
        Rsky1 = np.maximum(
            0.01, (0.13363 + 0.00077358 * alpha1 + beta1 * x0) / (1.0 + beta1 * x1)
        )
        beta_mean = 0.5 * (beta1 + beta2)
        r1a0 = rational([0.022550, 0.27375, -0.41817, 1.0, 4.5555, 0.0], beta_mean)
        r1a1 = rational([0.044773, 0.15345, -0.30157, 1.0, 3.4223, 0.0], beta_mean)
        am6 = np.minimum(6.0, mR)
        Rsky1 = Rsky1 * (1.0 + r1a0 * (am6 - 1.0)) / (1.0 + r1a1 * am6)
        # Sky albedo for backscattered radiation, Band 2, Appendix 1
        y0 = rational([0.14618, 0.062758, 0.0, 1.0, -0.19402, 0.0], alpha2)
        y1 = rational([0.58101, 0.174260, 0.0, 1.0, -0.17586, 0.0], alpha2)
        Rsky2 = np.maximum(
            0.005, (0.010191 + 0.00085547 * alpha2 + beta2 * y0) / (1.0 + beta2 * y1)
        )

        # Band 1 + Band 2
        Edif1 + Edif2
        Edir1 + Edir2
        rog = 0.5 * (albedo1 + albedo2)
        r1 = Rsky1 * rog
        edifd1 = r1 * (Edir1 + Edif1) / (1.0 - r1)
        edif1 = edifd1 + Edif1
        edif1[sza > 87.0] = np.maximum(
            0.2, edif1[sza > 87.0]
        )  # if(z.gt.87.)Edif1=Max(Edif1,.2) en el codigo fortran
        etot1 = edif1 + Edir1
        r2 = Rsky2 * rog
        edifd2 = r2 * (Edir2 + Edif2) / (1.0 - r2)
        edif2 = edifd2 + Edif2
        edif2[sza > 87.0] = np.maximum(
            0.1, edif2[sza > 87.0]
        )  # if(z.gt.87.)Edif2=Max(Edif2,.1) en el codigo fortran
        etot2 = edif2 + Edir2

        constraint = lambda value: np.where(is_day, value, 0.0)
        self.dni1 = constraint(Ebn1)
        self.dni2 = constraint(Ebn2)
        self.dni = constraint(Ebn1 + Ebn2)
        self.dir1 = constraint(Edir1)
        self.dir2 = constraint(Edir2)
        self.dir = constraint(Edir1 + Edir2)
        self.dif1 = constraint(edif1)
        self.dif2 = constraint(edif2)
        self.dif = constraint(edif1 + edif2)
        self.ghi1 = constraint(etot1)
        self.ghi2 = constraint(etot2)
        self.ghi = constraint(etot1 + etot2)


class JARARIAS(object):
    def __init__(self, ghi, sun, ma=False):
        """
        It calculates the diffuse fraction (kd) based on the regressive model proposed by Ruiz-Arias
        et al., 2010. Returned value is an object whose attributes are: kd, ghi, dni, dif
        > sun [SUN object]: position of the Sun.
        > ghi   [sequence]: accordingly with sun. It is used to derive the values of the radiation
                            components, returned as attributes of the returned object:
                            jararias.ghi,jararias.dhi(horizontal projected), jararias.dif, jararias.kd,
                            jararias.dni (normal)
        > ma     [boolean]: If False (default) the simplest model is used (model G0 in Ruiz-Arias et al., 2010).
                            Else a more complex model (which include ma, denoted as model G2 in Ruiz-Arias et
                            al., 2010) is used.
        NOTE: this model was properly adjusted for hourly values.
        """
        _limit = {
            "kt": (0.0, 1.0),
            "air_mass": (1.0e-6, 40.0),
            "cosZ": (1.0e-6, 1.0),
            "ghi": (0.0, None),
        }
        _dtype = np.float64

        kt = np.clip(
            np.array(sun.get_kt(ghi), dtype=_dtype), _limit["kt"][0], _limit["kt"][1]
        )
        self.ghi = _checkinput(ghi, "ghi", _limit["ghi"], kt, dtype=_dtype)
        if ma:
            ma = np.clip(
                np.array(sun.get_ma(), dtype=_dtype),
                _limit["air_mass"][0],
                _limit["air_mass"][1],
            )
            self.kd = 0.944 - (
                1.538
                * np.exp(
                    -1.0
                    * np.exp(
                        2.808
                        - (5.759 * kt)
                        + (2.276 * (kt**2))
                        - (0.125 * ma)
                        + (0.013 * (ma**2))
                    )
                )
            )
        else:
            self.kd = 0.952 - (1.041 * np.exp(-1.0 * np.exp(2.3 - (4.702 * kt))))
        cosz = np.clip(
            np.array(sun.get_csza(), dtype=_dtype), _limit["cosZ"][0], _limit["cosZ"][1]
        )
        self.dif = self.kd * self.ghi
        self.dni = (self.ghi - self.dif) / cosz


class DIRINT(object):
    def __init__(self, ghi, sun):
        """
        It returns an object whose attributes are the surface irradiance componets:
            + ghi (sattelite meassured; input)
            + dni (derived from DIRINT model (Perez, 1992), GHI to DNI model.
        Input values are:
        > ghi    [sequence]: float values of the GHI retrieved from the satellite platform. Missing admited values: NaN or -999.0
        > sun      [object]: SUN object (position of the Sun).
        NOTE: DIRINT function: DIRINT( G, Z, TD, ALT ,I0 )
        GUARNING: becareful with DIRINT option and the missing values. This funciton works with -999.0 for missing values.
        The best is to ensure that ghi values are properly filtered before to pass to the DIRINT.
        GUARNING: any GHI < 0.0 value will be automatically converted to a missing value.
        """
        _limit = {"ghi": (0.0, None), "cosZ": (0.0, 1.0)}
        _dtype = np.float64

        esNan = False

        ghi = np.asarray(ghi)
        if np.any(np.isnan(ghi)):
            esNan = True
            ghi = np.where(np.isnan(ghi), -999.0, ghi)
        isMissing = ghi < 0.0

        self.ghi = _checkinput(ghi, "ghi", _limit["ghi"], sun.dates, dtype=_dtype)
        self.ghi = np.where(isMissing, -999.0, self.ghi)
        mu = _checkinput(sun.get_csza(), "cosZ", _limit["cosZ"], self.ghi, dtype=_dtype)

        from dirint import dirint

        length = len(ghi)
        ghi = np.insert(self.ghi, [0, length], -999.0)
        z = np.arccos(mu)
        z = np.insert(z, [0, length], -999.0)
        td = np.zeros(length) - 999.0
        alt = np.zeros(length) + sun._elev
        Io = 1366.1 * sun.get_eccentricity()
        dni = np.asarray(
            [
                dirint(
                    [ghi[i], ghi[i + 1], ghi[i + 2]],
                    [z[i], z[i + 1], z[i + 2]],
                    td[i],
                    alt[i],
                    Io[i],
                )
                for i in range(length)
            ]
        )
        #        self.dni = np.where(mu <= 1.0e-6, 0.001, dni)
        self.dni = np.where(mu < 0.0, 0.0, dni)
        if esNan:
            self.ghi = np.where(isMissing, np.nan, self.ghi)
            self.dni = np.where(isMissing, np.nan, self.dni)
        else:
            self.dni = np.where(isMissing, -999.0, self.dni)


class DIRINDEX(object):
    def __init__(self, ghi, sun, clearsky, method="dirint"):
        """
        It returns an object whose attributes are the surface irradiance componets:
            + ghi (sattelite meassured; input)
            + dni (derived from DRINDEX method usin the JARARIAS (Ruiz-Arias, 2010), or dirint (Perez, 1992) (Default), GHI to DNI model,
            and the Simplified Version of SOLIS clear sky model (depending on the introduced clearsky object)).
        Input values are:
        > ghi    [sequence]: float values of the GHI retrieved from the satellite platform. Missing admited values: NaN or -999.0
        > sun      [object]: SUN object (position of the Sun).
        > clearsky [object]: clear sky model object (e.g. svSolis). Attributes must be "ghi" and "dni"
        > method   [string]: whether to use jararias (DEFAULT) or dirint (dirint) GHI-to-DNI method.
        NOTE: DIRINT function: DIRINT( G, Z, TD, ALT ,I0 )
        GUARNING: becareful with DIRINT option and the missing values. This funciton works with -999.0 for missing values.
        The best is to ensure that ghi values are properly filtered before to pass to the DIRINDEX.
        GUARNING: any GHI < 0.0 value will be automatically converted to a missing value.
        """
        _limit = {"ghi": (0.0, None), "cosZ": (0.0, 1.0)}
        _dtype = np.float64

        esNan = False

        if (method != "jararias") and (method != "dirint"):
            raise ValueError(
                "\n\n!0_/\tmacagoen-ERROR <swsolib.DIRINDEX>: method %s is neither jararias nor dirint.\n\n"
                % method
            )
        ghi = np.asarray(ghi)
        if np.any(np.isnan(ghi)):
            esNan = True
            ghi = np.where(np.isnan(ghi), -999.0, ghi)
        isMissing = ghi < 0.0
        self.ghi = _checkinput(ghi, "ghi", _limit["ghi"], clearsky.ghi, dtype=_dtype)
        mu = _checkinput(sun.get_csza(), "cosZ", _limit["cosZ"], self.ghi, dtype=_dtype)
        if method == "jararias":
            self.ghi = np.where(isMissing, np.nan, self.ghi)
            dni1 = JARARIAS(ghi=self.ghi, sun=sun).dni
            dni2 = JARARIAS(ghi=clearsky.ghi, sun=sun).dni
        elif method == "dirint":
            from dirint import dirint

            self.ghi = np.where(isMissing, -999.0, self.ghi)
            length = len(ghi)
            ghi = np.insert(self.ghi, [0, length], -999.0)
            csghi = np.insert(clearsky.ghi, [0, length], -999.0)
            z = np.arccos(mu)
            z = np.insert(z, [0, length], -999.0)
            td = np.zeros(length) - 999.0
            alt = np.zeros(length) + sun._elev
            Io = 1366.1 * sun.get_eccentricity()
            dni1 = np.asarray(
                [
                    dirint(
                        [ghi[i], ghi[i + 1], ghi[i + 2]],
                        [z[i], z[i + 1], z[i + 2]],
                        td[i],
                        alt[i],
                        Io[i],
                    )
                    for i in range(length)
                ]
            )
            dni2 = np.asarray(
                [
                    dirint(
                        [csghi[i], csghi[i + 1], csghi[i + 2]],
                        [z[i], z[i + 1], z[i + 2]],
                        td[i],
                        alt[i],
                        Io[i],
                    )
                    for i in range(length)
                ]
            )
        else:
            raise ValueError(
                "\n\n!0_/\tmacagoen-ERROR <libSolartools.DIRINDEX>: specified method <%s>  is not <jararias> nor <dirint>\n\n"
                % method
            )

        #        isDirint = (dni1 >= 0.0) & (dni2 >= 0.0000001)
        isDirint = (dni1 >= 0.0) & (dni2 > 0.0)
        auxCoef = np.clip(dni1[isDirint] / dni2[isDirint], 0.0, 2.0)
        coef = np.zeros(len(dni2)) + auxCoef.min()  # np.mean(auxCoef)
        coef[isDirint] = auxCoef
        self.dni = clearsky.dni * coef
        self.dni1 = dni1
        if esNan:
            self.ghi = np.where(isMissing, np.nan, self.ghi)
            self.dni = np.where(isMissing, np.nan, self.dni)
            self.dni1 = np.where(isMissing, np.nan, self.dni1)
        else:
            self.dni = np.where(isMissing, -999.0, self.dni)
            self.dni1 = np.where(isMissing, -999.0, self.dni1)


class MMAC(object):
    def __init__(self, sun, p=1013.25, uo=0.35, rhog=0.2, w=1.0, alpha=1.3, beta=0.06):
        """
        It computes the irradiance components for a sun object. Returned value is an object which attributes are:
        _ghi, dni, _diff.
            Written by C. Gueymard
            Revised 25 Feb 2014
            Method based on Unsworth-Monteith turbidity and
            Paltridge & Platt's radiation model.
            Used by Canadian authors from McMaster University (hence "MAC"):
            Uboegbulam & Davies (1983), Freund (1983),
            Hay & Darby (1984), and McGuffie et al (1985).
            Only Hay & Darby gave an expression for TrR, but with a sign
            error and disappointing performance (diverges for m>11).
            Formulation rather used here is from Davies (1987).

            Ozone was fixed at 0.35 cm in the original publications, so this
            default value could be used here. In operational applications, however,
            a daily variable ozone is recommended instead.

            For more info: Gueymard (Solar Energy 2003), who named this model "Modified MAC".

        Input values are:
        > sun [object]: SUN object (position of the Sun).
        > p    [float]: station pressure [mb]. Defaulted to 1013.25 mb.
        > uo   [float]: reduced ozone vertical pathlength (atm-cm). Defaulted to 0.35 cm.
        > rhog [float]: ground albedo. Default value 0.2.
        > w    [float]: precipitable water [cm]. Default value 1.0 cm.
        > alpha[float]: Angstrom turbidity exponent. Default value 1.3
        > beta [float]: Angstrom turbidity coefficient (AOD at 1 micron). Default value 0.06
        GUARNING: _ghi and _diff are not very good in accuracy.
        """
        _dtype = np.float64

        ecc = np.array(sun.get_eccentricity(), dtype=_dtype)
        cosZ = np.clip(np.array(sun.get_csza(), dtype=_dtype), 1.0e-6, 1.0)

        En0 = 1353.0 * ecc
        presc = p / 1013.25
        omeg = 0.98
        amr = 35.0 / (1.0 + 1224.0 * cosZ**2) ** 0.5
        am = amr * presc
        w1 = 10.0 * w
        wvlef = 0.695 + amr * (0.0160 + 0.066 * beta / (0.7**alpha))
        tauA = beta / (wvlef**alpha)
        TrR = 1.0 / (
            1.0
            + np.exp(-2.12182 + 0.791532 * np.log(am) - 0.024761 * (np.log(am)) ** 2)
        )
        x = amr * uo * 10.0
        aov = 0.002118 * x / (1.0 + 0.0042 * x + 0.00000323 * (x**2))
        aou = 0.1082 * x / (1.0 + 13.86 * x) ** 0.805 + 0.00658 * x / (
            1.0 + (10.36 * x) ** 3
        )
        Tro = 1.0 - aov - aou
        wp = w1 * amr * (presc**0.75)
        aw = 0.29 * wp / ((1.0 + 14.15 * wp) ** 0.635 + 0.5925 * wp)
        Tabs = TrR * Tro - aw
        Tra = np.exp(-am * tauA)
        self.dni = np.where(cosZ < 1.0e-5, 0.0, En0 * Tra * Tabs)
        Eb = self.dni * cosZ

        Trab = 0.95**1.66
        Fa = 0.93 - 0.21 * np.log(amr)
        Fab = 0.93 - 0.21 * np.log(1.66)
        rhos = 0.0685 + (1.0 - Trab) * omeg * (1.0 - Fab)
        Ed0 = (
            En0
            * cosZ
            * (0.5 * Tro * Trab * (1.0 - TrR) + omeg * Fa * Tabs * (1.0 - Tra))
        )
        Edb = rhog * rhos * (Eb + Ed0) / (1.0 - rhog * rhos)
        self._dif = np.where(cosZ < 1.0e-5, 0.0, Ed0 + Edb)
        self._ghi = Eb + self._dif


class BIRD(object):
    """
    Bird and Hulstrom, 1981. (Model C in Iqbal 1983, pag.188). Broadband parametric clear-sky model.
    It returns an object whose attributes are the ghi, dni and dif values.
    > sun [object]: SUN object (position of the Sun).
    > O3 [float]  : vertical O3 layer thickness
    > w [float]   : precipitable water
    > p [float]   : pressure at ground level (default 1000mba)
    > t [float]   : temperature at ground level (default 298K)

    """

    def __init__(self, sun, O3=0.35, w=1.4, p=1000.0, t=298.0):
        _limit = {
            "precipitable water": (0.2, 10.0),  # cm
            "pressure": (850.0, 1020.0),
        }  # hPa, limites que he puesto yo
        _dtype = np.float64

        # Paramters
        po = 1013.25  # hPa
        cosZ = np.array(sun.get_csza(), dtype=_dtype)
        ecc = np.array(sun.get_eccentricity(), dtype=_dtype)
        w = _checkinput(
            w, "precipitable water", _limit["precipitable water"], cosZ, dtype=_dtype
        )
        p = _checkinput(p, "pressure", _limit["pressure"], cosZ, dtype=_dtype)
        rhog = 0.2
        k038 = 0.087
        k05 = 0.069
        w0 = 0.9
        Fc = 0.84
        ma = sun.get_ma()
        mr = ma * np.exp(sun._elev / 8434.5)
        Io = 1366.1 * sun.get_eccentricity()

        is_day = cosZ > 1e-2

        # DNI
        ## Rayleigh
        tauR = np.exp(-0.0903 * (ma**0.84) * (1.0 + ma - ma**1.01))
        ## O3
        U3 = O3 * mr
        tauO = 1.0 - (
            0.1611 * U3 * ((1.0 + 139.48 * U3) ** -0.3035)
            - 0.002715 * U3 * ((1.0 + 0.044 * U3 + 0.0003 * (U3) ** 2) ** -1)
        )
        ## Mixed gases
        tauG = np.exp(-0.0127 * (ma**0.26))
        ## Water vapor
        w = w * ((p / po) ** (3.0 / 4.0)) * ((273.0 / t) ** 0.5)
        U1 = w * mr
        tauW = 1.0 - 2.4959 * U1 * (((1.0 + 79.034 * U1) ** 0.6828 + 6.385 * U1) ** -1)
        ## Aerosol
        ka = 0.2758 * k038 + 0.35 * k05
        tauA = np.exp(-(ka**0.873) * (1.0 + ka - ka**0.7088) * (ma**0.9108))

        self.dni = np.zeros(cosZ.shape)
        self.dni[is_day] = (
            0.975
            * Io[is_day]
            * tauR[is_day]
            * tauO[is_day]
            * tauG[is_day]
            * tauW[is_day]
            * tauA[is_day]
        )
        dhi = np.zeros(cosZ.shape)
        dhi[is_day] = self.dni[is_day] * cosZ[is_day]

        # Dif
        ## Rayleigh y Aerosol
        tauAA = 1.0 - (1 - w0) * (1 - ma + ma**1.06) * (1 - tauA)
        tauAS = tauA / tauAA
        difR_difA = (
            (0.79 * Io * cosZ * tauO * tauG * tauW * tauAA) / (1 - ma + ma**1.02)
        ) * (0.5 * (1 - tauR) + Fc * (1 - tauAS))
        ## Multiple refletion
        rhoa = 0.0685 + (1 - Fc) * (1 - tauAS)
        difM = ((dhi + difR_difA) * (rhog * rhoa)) / (1 - rhog * rhoa)

        self.dif = np.zeros(cosZ.shape)
        self.dif[is_day] = difR_difA[is_day] + difM[is_day]

        # GHI
        self.ghi = dhi + self.dif


# class CLSKBox(object)     : pass


class MesorBIRD(object):
    """
    Returns a CLSKBox object,whose attributes are the components of the clear sky model of Bird
    (described as Model C in Iqbal 1983), with almost no extintion. Attributes .ghi,.dni,.dif, are numpy arrays.
    > dates [list or numpy array]: date-times of interest, in datetime or float format, at which
                                    cosTheta is calculated. It is used only to obtain the I0 values,
                                    I0 = 1366.1(W/m2) * ecf
    > cosTheta [list or numpy array]: float cosine values of the zenith angle of the time
    > ecf [list or numpy array]: eccentricity_correction_factor for those dates
    > z [float]: elevation (in m AMSL) of the point of interest. By default at mean sea level (0 m)
    GUARNING: remember that cosTheta for hourly values could be the mean hourly value (the
    value at the the midle of the hour)
    GUARNING: values are returned in Wm-2
    NOTE: Solar radiation is "represented" in the time reference of cosTheta, which is supposed to
    correspond to dates. That is, becareful in the calculation of cosTheta: time reference of dates
    should be taken into account!
    """

    def __init__(self, sun):
        _dtype = np.float64

        ecc = np.array(sun.get_eccentricity(), dtype=_dtype)
        cosZ = np.clip(np.array(sun.get_csza(), dtype=_dtype), 0.0, 1.0)
        # cosZ = np.clip(np.array(sun.get_csza(),dtype=_dtype),1.0e-6,1.0)
        z = sun._elev

        # if ((not isST(cosTheta)) or (not isST(dates))) : raise ValueError('\nmacagoen-ERROR <libSolartools.clsk>: input values neither list nor numpy array ...\n')

        # cosZ           = np.asarray(cosTheta)
        # cosZ           = np.clip(cosZ,-1.,1.)
        I0 = 1366.1 * ecc

        rhoGround = 0.2
        rhoSky = 0.0685
        ma = np.clip(
            np.exp(-z / 8446.0)
            / (cosZ + 0.15 * ((93.885 - np.rad2deg(np.arccos(cosZ))) ** (-1.253))),
            0.0,
            60.0,
        )
        tauRay = np.exp((-0.0903 * (ma**0.84)) * (1.0 + ma - (ma**1.01)))
        dniClear = np.maximum(0.0, 0.9751 * tauRay * I0)
        difRay = 0.79 * I0 * cosZ * ((0.5 * (1 - tauRay)) / (1.0 - ma + ma**1.02))
        difMultiReflec = ((dniClear * cosZ) + difRay) * (
            (rhoGround * rhoSky) / (1 - (rhoGround * rhoSky))
        )
        difClear = np.maximum(0.0, difRay + difMultiReflec)
        ghiClear = np.maximum(0.0, dniClear * cosZ + difClear)

        # cs     = CLSKBox()
        self.ghi = np.where(cosZ <= 1.0e-5, 0.0, ghiClear)  # 90.0 deg
        self.dni = np.where(cosZ <= 0.05, 0.0, dniClear)  # 85.0 deg 0.08716
        self.dif = np.where(cosZ <= 1.0e-5, 0.0, difClear)  # 90.0 deg
        # return cs


class clear_sky_msg(object):
    """
    Entradas:
    sun = objeto SUN devuelto por solar.py
    lat = latitud en radianes
    lon = longitud en radianes
    elev = elevación en metros

    Salidas:
    G0 = GHI de cielo ultra-claro
    B0 = DNI de cielo ultra-claro
    """

    def __init__(self, sun):
        latitud = np.deg2rad(sun._lat)
        # lon = np.deg2rad(sun._lon)
        elev = sun._elev
        I0 = 1367.0  # constante solar
        declination = np.deg2rad(sun.get_declination())  # declinación en grados
        eccentricity = sun.get_eccentricity()  # excentricidad
        ma = sun.get_ma()  # masa óptica
        tst = sun.get_tst()  # tiempos solares

        # GHI (Wh/m²)
        Dl = 24 / 0.25
        I0e = I0 * eccentricity * Dl / (2 * np.pi)
        sins = np.sin(latitud) * np.sin(declination)
        coss = np.cos(latitud) * np.cos(declination)
        t2 = tst + 0.25 / 2
        t1 = t2 - 0.25
        omega1 = (t1 - 12.0) * np.pi / 12.0
        omega2 = (t2 - 12.0) * np.pi / 12.0
        G0 = I0e * (
            sins * (omega2 - omega1) + coss * (np.sin(omega2) - np.sin(omega1))
        )  # GHI
        self.ghi = np.clip(G0, 0, I0)

        # DNI (W/m²)
        elev_corr = np.clip(np.exp(-elev / 8434.5), 0.5, 1.0)  # altura corregida
        x = np.array([0.5, 0.75, 1.0])
        y = np.array(
            [
                1.68219 - 0.03059 * ma + 0.00089 * ma**2,
                1.248174 - 0.011997 * ma + 0.00037 * ma**2,
                1.0,
            ]
        )
        p_corr = np.interp(elev_corr, x, y)  # presión interpolada
        deltaR0 = (
            1.0
            / (
                6.625928
                + 1.92969 * ma
                - 0.170073 * ma**2
                + 0.011517 * ma**3
                - 0.000285 * ma**4
            )
        ) / p_corr
        deltaR1 = 1.0 / (10.4 + 0.718 * m)
        deltaR[ma <= 20] = deltaR0[ma <= 20]
        deltaR[ma > 20] = deltaR1[ma > 20]
        deltaR[ma <= 0] = 0.0  # espesor óptico de Rayleigh
        B0 = I0 * eccentricity * np.exp(-0.8662 * ma * deltaR)  # DNI
        self.dni = np.clip(B0, 0, I0)
        # self.ghi = G0
        # self.dni = B0


class projIrrad(object):
    """
    Irradiances Over Surfaces.
    Calcula la irradiancia global para tres tipos de seguimiento: fijo (0), 1 eje (1), 2 ejes (2).
    Si las irradiancias directa y/o difusa no se proporcionan se derivaran de la ghi mediante el
    cálculo de la fraccion de difusa, modelo Ruiz-Arias et al. 2010. Atributos del objeto son:
        + .mu0  : coseno angulo cenital (tomado del objeto sun)
        + .theta: angulo cenital [rad] (tomado del objeto sun)
        + .Ioh: irradiancia solar extraterrestre proyectada sobre superficie horizontal
        + .latitude: latitud [rad] (tomada del objeto sun)
        + .inclinacion: inclinacion [rad]
        + .orientacion: orientacion [rad]
        + .w: angulo horario [rad] (adaptado al criterio de signos de Iqbal)
        + .azimut: angulo azimutal [rad] (adaptado al criterio de signos de Iqbal)
        + .de: declinacion [rad] (adaptada al criterio de signos de Iqbal)
        + .ghi: irradiancia global horizontal
        + .bni: irradiancia directa (beam) normal (mismas unidades que ghi)
        + .bhi: irradiancia directa (beam) proyectada horizontal (mismas unidades que ghi)
        + .dhi: irradiancia difusa horizontal (mismas unidades que ghi)
        + .gsi0, .gsi1, .gsi2: irradiancia global proyectada sobre superficie
                               seguimiento tipo 0, 1 y 2
                               (mismas unidades que ghi)
        + .bsi0, .bsi1, .bsi2: irradiancia directa (beam) proyectada sobre superficie
                               seguimiento tipo 0, 1 y 2
                               (mismas unidades que ghi)
        + .dsi0, .dsi1, .dsi2: irradiancia difusa proyectada sobre superficie
                               seguimiento tipo 0, 1 y 2
                               (mismas unidades que ghi)
        + .kt: indice de claridad
        + .ALBEDO_SFC: albedo de superficie (constate, valor por defecto 0.2)
        + .mu_ax0, .mu_ax1, .mu_ax2: coseno del ángulo cenital solar en superficie inclinada
                                     seguimiento tipo 0, 1 y 2
    Constructor:
    > sun [object]: posicion del sol.
    > ghi [sequence]: irradiancia global horizontal. Sus unidades serán las unidades de las componentes
                      calculadas (normalmente [W/m2])
    > bni [sequence]: irradiancia directa normal. Unidades de ghi
    > dhi [sequence]: irradiancia difusa horizontal. Unidades de ghi
    > inclinacion [float]: por defecto se tomara el valor de la latitud - 5grados.
    > orientacion [float]: por defecto 0. Azimuth sign criteria: E > 0 / S = 0 / W < 0
    """

    def __init__(
        self, sun, ghi, bni=None, dhi=None, inclinacion=None, orientacion=None
    ):
        _limit = {
            "ghi": (0.0, None),
            "bni": (0.0, None),
            "dhi": (0.0, None),
            "cosZ": (0.0, 1.0),
        }
        _dtype = np.float64

        if np.any(bni) != np.any(dhi):
            raise ValueError(
                "\n\n!0_/\tmacagoen-ERROR <swsolib.swsGI>: if bni/dhi is not None then bni/dhi must be so.\n\n"
            )
        self.mu0 = _checkinput(
            sun.get_csza(), "cosZ", _limit["cosZ"], sun._dtnum, dtype=_dtype
        )
        self.theta = np.deg2rad(sun.get_sza())
        ecf = np.array(sun.get_eccentricity(), dtype=_dtype)
        Io = 1366.1 * ecf
        self.Ioh = np.clip(Io * self.mu0, 1e-4, 2000.0)
        self.latitude = np.radians(sun._lat)
        if inclinacion is None:
            self.inclinacion = np.radians(sun._lat - 5.0)
        else:
            self.inclinacion = np.radians(inclinacion)
        if orientacion is None:
            self.orientacion = 0.0
        else:
            self.orientacion = np.radians(orientacion)
        self.w = -1.0 * np.deg2rad(sun.get_hour_angle())
        self.azimut = np.deg2rad(180.0 - sun.get_saa())
        self.de = np.deg2rad(sun.get_declination())
        self.ghi = _checkinput(ghi, "ghi", _limit["ghi"], sun._dtnum, dtype=_dtype)
        if np.any(bni):
            self.bni = _checkinput(bni, "bni", _limit["bni"], sun._dtnum, dtype=_dtype)
            self.bhi = _checkinput(
                bni * self.mu0, "bni", _limit["bni"], sun._dtnum, dtype=_dtype
            )
        else:
            self.bni, self.bhi = None, None
        if np.any(dhi):
            self.dhi = _checkinput(dhi, "dhi", _limit["dhi"], sun._dtnum, dtype=_dtype)
        else:
            self.dhi = None
        self.kt = sun.get_kt(self.ghi)

        # Calculo componentes directa y difusa a partir de ghi (fraccion de difusa. Ruiz-Arias et al. 2010)
        if not (np.any(self.bni)):
            kd = 0.952 - 1.041 * np.exp(-np.exp(2.300 - 4.702 * self.kt))
            self.dhi = (
                self.ghi * kd
            )  # radiación instantánea difusa en superficie horizontal
            self.bhi = (
                self.ghi - self.dhi
            )  # radiación instantánea directa en superficie horizontal
            self.bni = self.bhi * self.mu0  # radiación instantánea directa normal
        self.ghi = np.where(self.mu0 < 1e-2, 0.0, self.ghi)
        self.dhi = np.where(self.mu0 < 1e-2, 0.0, self.dhi)
        self.bhi = np.where(self.mu0 < 1e-2, 0.0, self.bhi)
        self.bni = np.where(self.mu0 < 1e-2, 0.0, self.bni)
        self.kt = np.where(self.mu0 < 1e-2, 0.0, self.kt)

        self.gsi0, self.dsi0, self.bsi0 = (
            None,
            None,
            None,
        )  # Global, Diffuse and Beam Irradiances over the Surface
        self.gsi1, self.dsi1, self.bsi1 = None, None, None
        self.gsi2, self.dsi2, self.bsi2 = None, None, None

        self.ALBEDO_SFC = 0.2

    def _get_mu(self, beta, gamma):
        """
        Devuelve el coseno del ángulo cenital solar en superficie inclinada, radianes
        (Eq (1.6.5a) Iqbal)
        """
        mu = (
            (
                np.sin(self.latitude) * np.cos(beta)
                - np.cos(self.latitude) * np.sin(beta) * np.cos(gamma)
            )
            * np.sin(self.de)
            + (
                np.cos(self.latitude) * np.cos(beta)
                + np.sin(self.latitude) * np.sin(beta) * np.cos(gamma)
            )
            * np.cos(self.de)
            * np.cos(self.w)
            + np.cos(self.de) * np.sin(beta) * np.sin(gamma) * np.sin(self.w)
        )
        return np.clip(mu, 0.0, 1.0)

    def get_rad(self, seguimiento):
        """
        Proyeccion sobre superficies inclinadas: instalaciones fijas
        > seguimiento [int]: tipo de seguimiento:
            + fijo   = 0
            + 1 eje  = 1
            + 2 ejes = 2
        """
        if seguimiento == 0:
            beta = self.inclinacion
            gamma = self.orientacion
            mu = self._get_mu(beta, gamma)
            self.mu_ax0 = mu
        elif seguimiento == 1:
            beta = self.inclinacion
            gamma = self.azimut
            mu = self._get_mu(beta, gamma)
            self.mu_ax1 = mu
        elif seguimiento == 2:
            beta = self.theta
            mu = 1.0
            self.mu_ax2 = mu
        # Radiacion directa
        rb = mu / np.clip(self.mu0, 1e-4, 1.0)
        bsi = self.bhi * rb  # Eq (11.2.5), Iqbal, W/m2
        # Radiacion difusa (modelo anisotrópico de Hay)
        transmittance = self.bhi / self.Ioh
        dsi = self.dhi * (
            transmittance * rb + (1 - transmittance) * 0.5 * (1.0 + np.cos(beta))
        )  # Eq (11.5.8), Iqbal
        # Radiacion reflejada (modelo isotropico)
        ir = (
            self.ghi * self.ALBEDO_SFC * 0.5 * (1.0 - np.cos(beta))
        )  # Eq (11.3.3) Iqbal, W/m2
        # Radiacion global
        gsi = np.clip(dsi + bsi + ir, 0.0, 2000.0)
        # Limites
        gsi = np.where(self.mu0 < 1e-2, 0.0, gsi)
        dsi = np.where(self.mu0 < 1e-2, 0.0, dsi)
        bsi = np.where(self.mu0 < 1e-2, 0.0, bsi)
        # Asignacion
        if seguimiento == 0:
            self.gsi0, self.dsi0, self.bsi0 = gsi, dsi, bsi
        elif seguimiento == 1:
            self.gsi1, self.dsi1, self.bsi1 = gsi, dsi, bsi
        elif seguimiento == 2:
            self.gsi2, self.dsi2, self.bsi2 = gsi, dsi, bsi


# log('generando posiciones del dia ...')
# horizon = np.where(np.diff(np.sign(sol_5.get_csza())))[0]
# midday = np.diff(np.sign(np.diff(sol_5.get_csza())))
# midday = np.where(midday>=0,0,1)
# midday = np.where(midday)[0]
# log(' OK\n')

# def gti(ghi,mu,dni,dif,angle=None):
#'''
# It generates a tGNI (tracking Global Tilted Irradiance) time series based on GHI values
# and the models of Jararias (diffuse fraction for diffuse horizontal irradiance from GHI values)
# and Perez 1990 (for diffuse tilted irradiance).
# > ghi
# GUARNIGN: limited to zenith angles bellow 85º
#'''
#### Constants
# cos85 = 0.0872
# def epsilonLimits(e) :
# if   ((1     <= e) and (e < 1.065)) : return 1
# elif ((1.065 <= e) and (e < 1.230)) : return 2
# elif ((1.230 <= e) and (e < 1.500)) : return 3
# elif ((1.500 <= e) and (e < 1.950)) : return 4
# elif ((1.950 <= e) and (e < 2.800)) : return 5
# elif ((2.800 <= e) and (e < 4.500)) : return 6
# elif ((4.500 <= e) and (e < 6.200)) : return 7
# elif  (6.200 <= e)                  : return 8
# else                                : return 'Nan'

#### Correspondence of the tuple values with de F coefficients is :
##   (F11, F12, F13, F21, F22, F23). Keys 1 to 8 are the
##   the categories of the epsilon parameter.
# fCoef = {
# 1:(-0.008, 0.588,-0.062,-0.060, 0.072,-0.022),
# 2:( 0.130, 0.683,-0.151,-0.019, 0.066,-0.029),
# 3:( 0.330, 0.487,-0.221, 0.055,-0.064,-0.026),
# 4:( 0.568, 0.187,-0.295, 0.109,-0.152,-0.014),
# 5:( 0.873,-0.392,-0.362, 0.226,-0.462, 0.001),
# 6:( 1.132,-1.237,-0.412, 0.288,-0.823, 0.056),
# 7:( 1.060,-1.600,-0.359, 0.264,-1.127, 0.131),
# 8:( 0.678,-0.327,-0.250, 0.156,-1.377, 0.251),
#'Nan':(np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)}

#### Values of GHI, DNI, DIF, zenith and cosZ, I0, ma (optical air mass)
# zenith = np.cos(mu)                        #Solar zenith angle (in radians)
# cosZ   = mu                                 # cosine of zenith
# up85   = (cos85 <= cosZ)
# cosZ   = np.where(up85, cosZ, np.nan)
# zenith = np.where(np.isfinite(cosZ),zenith,np.nan)
# kt     = kt(,mdfGHI.ghi,cosZ,aiKt='Kasten',elevation=mdfGHI.get_elevation())  # clearness index
# kt     = np.where(kt <= 1.,kt,1.)
# kt     = np.where(up85,kt,np.nan)
# mod    = st.jararias(kt,ghi=mdfGHI.ghi)          # diffuse fraction
# dif    = mod.dif
# dni    = mod.dhi / cosZ
# ma     = st.ma(cosZ,mdfGHI.get_elevation())   # optical relative air mass
# I0     = st.ecf(mdfGHI.times)*1366.1          # extraterrestrial Irradiance: solar constant * eccentricity correction factor

#### Perez's model
# delta   = (dif * ma) / I0                                               # Delta parameter
# epsilon = (((dif+dni)/dif)+1.041*(zenith**3))/(1+1.041*(zenith**3))     # Epsilon values
# epsCat  = map(epsilonLimits,epsilon)                                    # Epsilon categories
# F1      = lambda f11,f12,f13,d,z : f11 + f12*d + f13*z                  # F1 coefficient
# F2      = lambda f21,f22,f23,d,z : f21 + f22*d + f23*z                  # F2 coefficient
# a       = 1.                                                            # For tracking surfaces: max(0,cosTheta), Theta = 0
# b       = np.where(up85, cosZ, cos85)                                   # max(0.087,cosZ) => values of Z bellow 85º
##~ difn    = []
##~ for i,ec in enumerate(epsCat) :
##~ factor1 = 0.5 * ((1-F1(fCoef[ec][0],fCoef[ec][1],fCoef[ec][2],delta[i],zenith[i]))*(1+cosZ[i]))
##~ factor2 = F1(fCoef[ec][0],fCoef[ec][1],fCoef[ec][2],delta[i],zenith[i]) * (a/b[i])
##~ factor3 = F2(fCoef[ec][3],fCoef[ec][4],fCoef[ec][5],delta[i],zenith[i]) * np.sin(zenith[i])
##~ print factor1,factor2,factor3
##~ difn.append(factor1 + factor2 + factor3)
##~ difn = dif*np.asarray(difn)

# difn    = dif* np.asarray([((((1-F1(fCoef[ec][0],fCoef[ec][1],fCoef[ec][2],delta[i],zenith[i]))*(1+cosZ[i])) / 2)
# + F1(fCoef[ec][0],fCoef[ec][1],fCoef[ec][2],delta[i],zenith[i]) * (a/b[i])
# + F2(fCoef[ec][3],fCoef[ec][4],fCoef[ec][5],delta[i],zenith[i]) * np.sin(zenith[i]))
# for i,ec in enumerate(epsCat)])
# gni     = dni + difn + mdfGHI.ghi*((1-cosZ)/2)*0.128                    # considered an albedo of 0.128


class powerPV(object):
    """
    Devuelve un objeto cuyo atributo (epv) es el valor de la energia PV producida por la instalacion especificada
    para los valores de irradiancia global (GI) introducidos.
    PV-Power. Modelo PV.
    (VxD = Valor por Defecto)
    Devuelve Pgen [float]: POTENCIA generada "durante" GNI, en [kW]. Si GNI es horaria entonces Pgen es
                        igual a la ENERGÍA producida en esa hora, en kWh. La energía producida en Kwh
                        será Pgen si es en resolución horaria. Si es diezminutal hay que multiplicar
                        por 1/6 para obtener la ENERGÍA en kWh.
    Constructor:
    > GI [iterable] : serie temporal de irradiancia sobre la superficie horizontal o inclinada, [W/m2]
    > Tamb [iterable]: temperatura ambiente, [ºC]
    > Tstc [float]   : Temperature in Standard Test Conditions [ºC], VxD 25.0 ºC
    > cpt [float]    : coeficiente de pérdidas por temperatura (gamma). Depende del fabricante. VxD 0.004
    > nocT [float]   : Nominal Operation Cell Temperature, [ºC]. VxD = 47 ºC
    > total_power [float]  : potencia total de la instalación (potencia pico), [kW]
    NOTA: Los últimos valores son:
            - coeficiente de rendimiento por pérdidas de conducción
            - por el inversor.
    GUARNING: mucho cuidado con las unidades. Respetar las indicaciones: si por ejemplo Tstc se pide en ºC,
    pero internamente en el modelo hace falta en K, se hará la transformación de manera interna, pero el dato
    debe introducirse en ºC.
    """

    def __init__(self, GI, Tamb, total_power, Tstc=25.0, cpt=0.004, nocT=47.0):
        Isi = np.asarray(GI)
        Ta = np.asarray(Tamb) + 273.15
        Tcem = Tstc + 273.15
        gamma = cpt
        Tonc = nocT + 273.15
        Ppico = total_power
        Tc = Isi * (Tonc - 293.0) / 800.0 + Ta
        self.epv = ((Ppico * Isi * 0.001) * (1.0 - gamma * (Tc - Tcem))) * 0.97 * 0.95


class ESRA(object):
    """
    Returns a ESRABox object,whose attributes are the components of the clear sky model of ESRA (irradiance
    version) (Rigollier et al., 2000). Attributes .ghi,.dni (normal beam), .dhi (horizontal projection),
    .dif, are numpy arrays. Values are returned in Wm-2.
    > sun
    > tl [element or list or numpy array]: linke turbidity values at Air Mass = 2 at the location. Must be 12 values (monthly)
                                           [None,NaN,3.4,None,None,NaN,-999.9,'Value',None,None,None,None] is acceptable for
                                           a date in March
    """

    @staticmethod
    def expandTL(dates, TL):
        if isinstance(dates[0], float):
            dates = num2date(dates)
        else:
            pass
        if TL is None:
            TL = [2.3, 2.5, 2.4, 3.1, 3.4, 3.4, 3.4, 3.4, 3.3, 3.1, 2.7, 2.5]
        tl = np.asarray(TL)
        if len(tl) != 12:
            raise ValueError(
                "\n\n!0_/\tmacagoen-ERROR <swsolib.ESRA.expandTL>: TL values must be 12 (monthly values). Introduced: %d.\n\n"
                % len(tl)
            )
        itermonths = np.asarray([d.month for d in dates])
        return np.asarray([tl[i - 1] for i in itermonths])

    def __init__(self, sun, TL=None):
        _limit = {"cosZ": (0.0, 1.0)}

        _dtype = np.float64

        cosTheta = _checkinput(
            sun.get_csza(), "cosZ", _limit["cosZ"], sun._dtnum, dtype=_dtype
        )
        ecf = np.array(sun.get_eccentricity(), dtype=_dtype)
        eI0 = 1366.1 * ecf

        self.TL = self.expandTL(sun.dates, TL)

        a = np.arcsin(cosTheta)
        m = sun.get_ma(model="esra")
        deltaR = np.where(
            m <= 20.0,
            1.0
            / (
                6.6296
                + 1.7513 * m
                - 0.1202 * (m**2)
                + 0.0065 * (m**3)
                - 0.00013 * (m**4)
            ),
            1.0 / (10.4 + 0.718 * m),
        )
        self.dhi = np.where(
            cosTheta <= 0.0,
            0.0,
            eI0 * np.sin(a) * np.exp(-0.8662 * self.TL * m * deltaR),
        )  # 85.0 deg

        trd = -0.015843 + 0.030543 * self.TL + 0.0003797 * (self.TL**2)
        a0 = 0.26463 - 0.061581 * self.TL + 0.0031408 * (self.TL**2)
        a0 = np.where(a0 * trd < 0.002, 0.002 / trd, a0)
        a1 = 2.0402 + 0.018945 * self.TL - 0.011161 * (self.TL**2)
        a2 = -1.3025 + 0.039231 * self.TL + 0.0085079 * (self.TL**2)
        fd = a0 + a1 * np.sin(a) + a2 * (np.sin(a) ** 2)
        self.dif = np.where(cosTheta <= 0.0, 0.0, eI0 * trd * fd)  # 90.0 deg

        self.ghi = np.where(cosTheta <= 0.0, 0.0, self.dhi + self.dif)  # 90.0 deg
        cosZ = np.clip(cosTheta, 1.0e-10, 1.0)
        self.dni = np.where(cosTheta <= 0.0, 0.0, self.dhi / cosZ)


if __name__ == "__main__":

    from datetime import datetime, timedelta

    dates = pl.drange(
        datetime(2000, 6, 21, 0, 0), datetime(2000, 6, 22, 0, 0), timedelta(minutes=1)
    )
    sunpos = SUN(dates, -3.5, 37.5)
    solis = svSOLIS(sunpos)
    rest2 = REST2(sunpos)
    jararias = JARARIAS(solis.ghi, sunpos)
    DIRINDEX = DIRINDEX(solis.ghi, sunpos, solis, method="dirint")
    bird = BIRD(sunpos)
    fig = pl.figure(1)
    ax = fig.add_subplot(111)
    ax.plot_date(dates, rest2.dni, ls="-", marker="", color="r", label="REST2 DNI")
    ax.plot_date(dates, rest2.dif, ls="-", marker="", color="b", label="REST2 DIF")
    ax.plot_date(dates, rest2.ghi, ls="-", marker="", color="g", label="REST2 GHI")
    ax.plot_date(dates, solis.dni, ls="--", marker="", color="r", label="SOLIS DNI")
    ax.plot_date(
        dates, solis.dif, ls="--", marker="", color="b", label="SOLIS DIF (GHI-DIR)"
    )
    ax.plot_date(
        dates, solis.dif_orig, ls="-.", marker="", color="b", label="SOLIS DIF"
    )
    ax.plot_date(dates, solis.ghi, ls="--", marker="", color="g", label="SOLIS GHI")
    ax.plot_date(
        dates,
        jararias.dni,
        ls="-",
        marker="",
        color="orange",
        label="JARARIAS clear-sky DNI",
    )
    ax.plot_date(
        dates,
        DIRINDEX._DNI2,
        ls="-",
        marker="",
        color="brown",
        label="Dirint clear-sky DNI",
    )
    ax.plot_date(dates, bird.ghi, ls="-", marker="", color="#200053", label="BIRD GHI")
    ax.plot_date(dates, bird.dni, ls="-", marker="", color="#3AB23A", label="BIRD DNI")
    ax.plot_date(dates, bird.dif, ls="-", marker="", color="#DB6408", label="BIRD DIF")
    ax.legend(loc="upper left", fontsize=12)
    pl.show()
