import numpy as np

from astropy import units as u
from astropy import constants as const
from astropy.units import Quantity

# Type annotation
from typing import Any, Union, Optional, List, TypeVar




# Custom type
T = TypeVar("T", bound=Union[Quantity, np.ndarray, List[float], List[np.float64]])


def moment_of_inertia(M: T, R: T) -> Quantity:
    """
    RETURN
    ======
        - The moment of inertia for a solid homogeneous sphere; Units [Kg m^(2)]
    ARGS
    =====
        - M : the mass of the neutron star; Units [Msun]
        - R : the radius of the neutron star; Units [km]
    """
    M = u.Quantity(M, u.Msun)
    R = u.Quantity(R, u.km)

    return ((2./5.) * M * (R**2.)).decompose()

def magnetic_moment(B: Union[T, float], R: T) -> Quantity:
    """
    RETURN
    ======
        - The magnetic dipole moment; Units [A m^(2)]
    ARGS
    =====
        - B : the magnetic flux density; Units [G]
        - R : the radius of the neutron star; Units [km]
    """
    B = u.Quantity(B, u.gauss)
    R = u.Quantity(R, u.km)

    return ((4.*np.pi*B*(R**3.))/(const.mu0)).decompose()

def dipole_energy_loss(mu: T, fspin: T, i: float) -> Quantity:
    """
    RETURN
    ======
        - Loss of rotational energy due to emission of magnetodipole waves; Units [Kg m^(2) s^(-3)]
    ARGS
    =====
        - mu    : the magnetic dipole moment; Units [A m^(2)]
        - fspin : the spin frequency of the neutron star; Units [Hz]
        - i     : the incination angle; Units [rad]
    """
    fspin = u.Quantity(fspin, u.hertz)
    mu = u.Quantity(mu, u.ampere * (u.meter**2.))
    i = u.Quantity(i, u.rad)

    # The magnetic dipole moment is treated as a dimensionless quantity
    # For this reason, we multiply with mu0/4pi
    e = ((const.mu0/(4.*np.pi)) * (2.*(mu**2.)*((2.*np.pi*fspin)**4.)*(np.sin(i))**2.)/(3.*(const.c)**3.)).decompose()
    return e


def lc_radius(fspin: T) -> Quantity:
    """
    RETURN
    ======
        - The light cylinder radius; Units [m]
    ARGS
    =====
        - fspin : the spin frequency of the neutron star; Units [Hz]
    """
    fspin = u.Quantity(fspin, u.hertz)

    return (const.c / (2.*np.pi*fspin)).decompose()



def magnetic_radius(Mdot: T,
                    B: Union[T, float],
                    R: T,
                    M: T,
                    phi: float) -> Quantity:
    """
    RETURN
    ======
        - The magnetic radius; Units [m]
    ARGS
    =====
        - Mdot : the mass accretion rate; Units [Msun yr^(-1)]
        - B    : the magnetic flux density; Units [G]
        - R    : the radius of the neutron star; Units [km]
        - M    : the mass of the neutron star; Units [Msun]
        - phi  : a factor between 0.5 and 1.4
    """

    Mdot = u.Quantity(Mdot, u.M_sun / u.year)
    B = u.Quantity(B, u.gauss)
    R = u.Quantity(R, u.km)
    M = u.Quantity(M, u.Msun)
    phi = u.Quantity(phi, u.dimensionless_unscaled)

    alfven_radius = ((4. * np.pi / const.mu0) ** (2./7.) * (B**(4./7.)) * (R**(12./7.)) * \
                     ((Mdot * np.sqrt(2. * const.G * M))**(-2./7.))).decompose()

    magnetic_radius = phi * alfven_radius

    if magnetic_radius < R.to(u.meter):
        return R.to(u.meter)
    else:
        return magnetic_radius



def corotation_radius(fspin: T, M: T) -> Quantity:
    """
    RETURN
    ======
        - The corotation radius; Units [m]
    ARGS
    =====
        - fspin : the spin frequency of the neutron star; Units [Hz]
        - M : the mass of the neutron star; Units [Msun]
    """
    fspin = u.Quantity(fspin, u.hertz)
    M = u.Quantity(M, u.Msun)

    return (((2 * np.pi * fspin) ** (-2/3)) * ((const.G * M) ** (1/3))).decompose()



def fastness_parameter(r_mag: T, r_co: T) -> Quantity:
    """
    RETURN
    ======
        - The fastness parameter;
    ARGS
    =====
        - r_mag : the magnetic radius; Units [m]
        - r_co  : the corotation radius; Units [m]
    """
    r_mag = u.Quantity(r_mag, u.meter)
    r_co = u.Quantity(r_co, u.meter)

    return ((r_mag / r_co)**(3/2)).decompose()


def eq_spin_frequency(M:T, Rmag:T) -> Quantity:
    """
    RETURN
    ======
        - The equilibrium spin frequency; Units [Hz]
    ARGS
    =====
        - M        : the mass of the neutron star; Units [Msun]
        - Rmag     : the magneticradius of the neutron star; Units [km]
    """
    M = u.Quantity(M, u.Msun)
    Rmag = u.Quantity(Rmag, u.km)
    return (1/(2*np.pi)) * np.sqrt(const.G*M/(Rmag**3)).to(u.Hz)


def min_eq_spin_frequency(Mdot: T,
                    M: T,
                    R: T,
                    B: Union[T, float],
                    Mdot_edd: float = 1.5e-8) -> Quantity:
    """
    RETURN
    ======
        - The minimum equilibrium spin frequency; Units [Hz]
    ARGS
    =====
        - Mdot     : the mass accretion rate; Units [Msun yr^(-1)]
        - M        : the mass of the neutron star; Units [Msun]
        - R        : the radius of the neutron star; Units [km]
        - B        : the magnetic flux density; Units [G]
        - Mdot_edd : the mass accretion Eddington limit; Units [Msun yr^(-1)]
    """
    Mdot = u.Quantity(Mdot, u.Msun / u.yr)
    B = u.Quantity(B, u.gauss)
    Mdot_edd = u.Quantity(Mdot_edd, u.Msun / u.yr)
    M = u.Quantity(M, u.Msun)
    R = u.Quantity(R, u.km)

    pmin = 0.71 *u.ms * (B/(1e8*u.gauss)) ** (6./7.) * (Mdot /(0.1*Mdot_edd))**(-3./7.) * \
        (M/(1.4*u.Msun)) ** (-5./7.) * (R/(10*u.km))**(18./7.)

    return (1/pmin).to(u.Hz)



def accretion_torque(Mdot: T,
                    fspin: T,
                    B: Union[T, float],
                    M: T,
                    R: T,
                    i: float,
                    phi: float = 1.0,
                    xi: float = 1.0,
                    delta_omega: float = 0.002) -> Quantity:
    """
    RETURN
    ======
        - The accretion torque based on Tauris (2012)
        (https://arxiv.org/pdf/1202.0551.pdf); Units [kg m^(2) s^(-2)]
    ARGS
    =====
        - Mdot        : the mass accretion rate; Units [Msun yr^(-1)]
        - fspin       : the spin frequency of the neutron star; Units [Hz]
        - B           : the magnetic flux density; Units [G]
        - M           : the mass of the neutron star; Units [Msun]
        - R           : the radius of the neutron star; Units [km]
        - i           : the inclination angle; Units [rad]
        - phi         : a factor between 0.5 and 1.4 to calculate the magnetic radius
        - xi          : a factor ...
        - delta_omega : the width of transition zone near the magnetospheric boundary.
    """

    Mdot = u.Quantity(Mdot, u.Msun / u.yr)
    fspin = u.Quantity(fspin, u.hertz)
    B = u.Quantity(B, u.gauss)
    M = u.Quantity(M, u.Msun)
    R = u.Quantity(R, u.km)
    i = u.Quantity(i, u.rad)
    phi = u.Quantity(phi, u.dimensionless_unscaled)
    xi = u.Quantity(xi, u.dimensionless_unscaled)
    delta_omega = u.Quantity(delta_omega, u.dimensionless_unscaled)



    r_mag = magnetic_radius(Mdot=Mdot, B=B, R=R, M=M, phi=phi)
    r_co = corotation_radius(fspin=fspin, M=M)
    omega = fastness_parameter(r_mag=r_mag, r_co=r_co)
    mu = magnetic_moment(B=B, R=R)
    energy_loss = dipole_energy_loss(mu=mu, fspin=fspin, i=i)

    step_function = np.tanh(((1. - omega)/delta_omega) * u.rad)


    # Multiply 2nd term with mu0/4pi to make dipole moment
    # a dimensionless quantity. This results to N having units of [N m]
    N1 = (step_function * (Mdot * np.sqrt(const.G * M * r_mag) * xi)).decompose()
    N2 = (step_function * (const.mu0/(4.*np.pi)) * (mu**2. / (9.*(r_mag**3.)))).decompose()
    N3 = ((energy_loss)/(2.*np.pi*fspin)).decompose()

    N = N1 + N2 - N3

    return N


if __name__ == "__main__":
    pass