import numpy as np

from astropy import units as u
from astropy import constants as const
from astropy.units import Quantity

# Type annotation
from typing import Any, Union, Optional, List, TypeVar




# Custom type
T = TypeVar("T", bound=Union[Quantity, np.ndarray, List[float], List[np.float64]])


def a_posttrans(
    a_pretrans: T,
    delta_m: T,
    donor_mass_pretrans: T,
    ns_mass_pretrans: T,
    kick_velocity: T,
    theta: T,
) -> Quantity:
    """
    Gives the relation between the pre- and post-transition semimajor axes.

    Returns
    -------
    The post-transition semi-major axis. Units[AU]

    Args
    -------
    a_pretrans           : the pre-transition semi-major axis. Units [AU]
    delta_m              : the amount of gravitational mass-loss due to transitioning. Units[Msun]
    donor_mass_pretrans  : the mass of the donor star before the transition. Units [Msun]
    ns_mass_pretrans     : the mass of the neutron star before the transiton. Units [Msun]
    kick_velocity        : the magnitude of the kick velocity. Units [Km/s]
    theta                : the angle between the kick velocity vector and the
                           pre-transition orbital plane vector. Units [rad]
    """
    a_pretrans = u.Quantity(a_pretrans, u.AU)
    delta_m = u.Quantity(delta_m, u.Msun)
    donor_mass_pretrans = u.Quantity(donor_mass_pretrans, u.Msun)
    ns_mass_pretrans = u.Quantity(ns_mass_pretrans, u.Msun)
    kick_velocity = u.Quantity(kick_velocity, u.km / u.s)
    theta = u.Quantity(theta, u.rad)

    M = donor_mass_pretrans + ns_mass_pretrans
 
    # relative velocity between the two stars
    v_rel = np.sqrt(const.G * M / a_pretrans).to(
        u.km / u.s
    )  
    

    numerator = a_pretrans * (1 - (delta_m / M))
    denominator = (
        1
        - (2 * delta_m / M)
        - (kick_velocity / v_rel) ** 2
        - (2 * np.cos(theta) * (kick_velocity / v_rel))
    )

    return (numerator / denominator).to(u.AU)


def orbital_energy_posttrans(
    a_posttrans: T, donor_mass_posttrans: T, ns_mass_posttrans: T
) -> Quantity:
    """
    Gives the post-transition orbital energy.

    Returns
    -------
    The post-transition orbital energy. Units[J]

    Args
    -------
    a_posttrans            : the post-transition semi-major axis. Units[AU]
    donor_mass_posttrans   : the mass of the donor star after the transition. Units [Msun]
    ns_mass_posttrans      : the mass of the neutron star after the transiton. Units [Msun]
    """
    a_posttrans = u.Quantity(a_posttrans, u.AU)
    donor_mass_posttrans = u.Quantity(donor_mass_posttrans, u.Msun)
    ns_mass_posttrans = u.Quantity(ns_mass_posttrans, u.Msun)

    return -(const.G * (donor_mass_posttrans**1) * (ns_mass_posttrans**1)) / (
        2 * a_posttrans
    )


def orbital_period_posttrans(
    a_posttrans: T, donor_mass_posttrans: T, ns_mass_posttrans: T
) -> Quantity:
    """
    Kepler's 3rd law for the post-transition orbital period.

    Returns
    -------
    The post-transition orbital period. Units[days]

    Args
    -------
    a_posttrans            : the post-transition semi-major axis. Units[AU]
    donor_mass_posttrans   : the mass of the donor star after the transition. Units [Msun]
    ns_mass_posttrans      : the mass of the neutron star after the transiton. Units [Msun]
    """
    a_posttrans = u.Quantity(a_posttrans, u.AU)
    donor_mass_posttrans = u.Quantity(donor_mass_posttrans, u.Msun)
    ns_mass_posttrans = u.Quantity(ns_mass_posttrans, u.Msun)

    return (
        2
        * np.pi
        * np.sqrt(
            (a_posttrans**3) / (const.G * (donor_mass_posttrans + ns_mass_posttrans))
        ).to(u.d)
    )


def eccentricity_posttrans(
    a_pretrans: T,
    a_posttrans: T,
    donor_mass_posttrans: T,
    ns_mass_posttrans: T,
    kick_velocity: T,
    theta: T,
    phi: T,
) -> Quantity:
    """
    Gives the post-transition eccentricity.

    Returns
    -------
    The post-transition eccentricity.

    Args
    -------
    a_pretrans             : the pre-transition semi-major axis. Units [AU]
    a_posttrans            : the post-transition semi-major axis. Units[AU]
    donor_mass_posttrans   : the mass of the donor star after the transition. Units [Msun]
    ns_mass_posttrans      : the mass of the neutron star after the transiton. Units [Msun]
    kick_velocity          : the magnitude of the kick velocity. Units [Km/s]
    theta                  : the angle between the kick velocity vector and the
                             pre-transition orbital plane vector. Units [rad]
    phi                    : the kick angle on the plane perpendicular to the pre-transition orbit. Units[rad]
    """
    a_pretrans = u.Quantity(a_pretrans, u.AU)
    a_posttrans = u.Quantity(a_posttrans, u.AU)
    donor_mass_posttrans = u.Quantity(donor_mass_posttrans, u.Msun)
    ns_mass_posttrans = u.Quantity(ns_mass_posttrans, u.Msun)
    kick_velocity = u.Quantity(kick_velocity, u.km / u.s)
    theta = u.Quantity(theta, u.rad)
    phi = u.Quantity(phi, u.rad)

    #  post-transition reduced mass
    mu = 1 / (
        (1 / donor_mass_posttrans) + (1 / ns_mass_posttrans)
    )  
    

    M = donor_mass_posttrans + ns_mass_posttrans
    
    # relative velocity between the two stars
    v_rel = np.sqrt(const.G * M / a_pretrans).to(u.km / u.s)
    L_orb = (
        a_pretrans
        * mu
        * np.sqrt(
            (v_rel + (kick_velocity * np.cos(theta))) ** 2
            + (kick_velocity * np.sin(theta) * np.sin(phi)) ** 2
        )
    )

    orbital_energy = orbital_energy_posttrans(
        a_posttrans, donor_mass_posttrans, ns_mass_posttrans
    )

    return np.sqrt(
        1
        + (
            (2 * orbital_energy * (L_orb**2))
            / (
                mu
                * (const.G**2)
                * (donor_mass_posttrans**2)
                * (ns_mass_posttrans**2)
            )
        )
    )




if __name__ == "__main__":
    pass