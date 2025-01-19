from math import log

G_0 = 9.81  # m/s^2, gravitational acceleration at Earth's surface

def thrust(m_dot: float, v_e: float, p_e: float, p_0: float, A_e: float) -> float:
    """
    Calculate the total thrust of a rocket engine.

    Parameters:
    - m_dot: Mass flow rate (kg/s)
    - v_e: Effective exhaust velocity (m/s)
    - p_e: Exit pressure (Pa)
    - p_0: Ambient pressure (Pa)
    - A_e: Nozzle exit area (m^2)

    Returns:
    - Thrust (N)
    """
    if p_e < p_0:
        raise ValueError("Exit pressure should not be less than ambient pressure for positive thrust contribution.")
    return m_dot * v_e + (p_e - p_0) * A_e

def exhaust_velocity(I_sp: float, g_0: float = G_0) -> float:
    """
    Calculate the effective exhaust velocity.

    Parameters:
    - I_sp: Specific impulse (s)
    - g_0: Gravitational acceleration (m/s^2, default is from module constant)

    Returns:
    - Effective exhaust velocity (m/s)
    """
    return I_sp * g_0

def delta_v(v_e: float, m_0: float, m_f: float) -> float:
    """
    Calculate the delta-v (velocity change) using the Tsiolkovsky rocket equation.

    Parameters:
    - v_e: Effective exhaust velocity (m/s)
    - m_0: Initial mass of the rocket (kg)
    - m_f: Final mass of the rocket (kg)

    Returns:
    - Delta-v (m/s)
    """
    if m_0 <= m_f:
        raise ValueError("Initial mass must be greater than final mass for positive delta-v.")
    return v_e * log(m_0 / m_f)

def specific_impulse(F: float, m_dot: float, g_0: float = G_0) -> float:
    """
    Calculate the specific impulse.

    Parameters:
    - F: Thrust (N)
    - m_dot: Mass flow rate (kg/s)
    - g_0: Gravitational acceleration (m/s^2, default is from module constant)

    Returns:
    - Specific impulse (s)
    """
    if m_dot == 0:
        raise ZeroDivisionError("Mass flow rate cannot be zero for specific impulse calculation.")
    return F / (m_dot * g_0)

def mass_flow_rate(rho_e: float, A_e: float, v_e: float) -> float:
    """
    Calculate the mass flow rate.

    Parameters:
    - rho_e: Exhaust gas density (kg/m^3)
    - A_e: Nozzle exit area (m^2)
    - v_e: Effective exhaust velocity (m/s)

    Returns:
    - Mass flow rate (kg/s)
    """
    return rho_e * A_e * v_e

def nozzle_area_ratio(A_e: float, A_t: float, gamma: float, M_e: float) -> float:
    """
    Calculate the nozzle area ratio.

    Parameters:
    - A_e: Exit area (m^2)
    - A_t: Throat area (m^2)
    - gamma: Specific heat ratio (dimensionless)
    - M_e: Mach number at the nozzle exit (dimensionless)

    Returns:
    - Area ratio (dimensionless)
    """
    term1 = (gamma + 1) / 2
    term2 = 1 + (gamma - 1) / 2 * M_e**2
    # Corrected formula for area ratio calculation:
    return (A_e / A_t) / ((term1**(term1 / (gamma - 1))) * (1 / M_e) * (term2**((gamma + 1) / (2 * (gamma - 1)))))

def thermal_efficiency(T_e: float, T_c: float) -> float:
    """
    Calculate the thermal efficiency.

    Parameters:
    - T_e: Exhaust gas temperature (K)
    - T_c: Combustion chamber temperature (K)

    Returns:
    - Thermal efficiency (dimensionless) - Simplified Carnot efficiency calculation
    """
    if T_e >= T_c:
        raise ValueError("Exhaust temperature cannot be greater than or equal to combustion chamber temperature.")
    return 1 - (T_e / T_c)

def characteristic_velocity(p_c: float, A_t: float, m_dot: float) -> float:
    """
    Calculate the characteristic velocity (c*).

    Parameters:
    - p_c: Combustion chamber pressure (Pa)
    - A_t: Nozzle throat area (m^2)
    - m_dot: Mass flow rate (kg/s)

    Returns:
    - Characteristic velocity (m/s)
    """
    if m_dot == 0:
        raise ZeroDivisionError("Mass flow rate cannot be zero for characteristic velocity calculation.")
    return (p_c * A_t) / m_dot

def thrust_coefficient(F: float, p_c: float, A_t: float) -> float:
    """
    Calculate the thrust coefficient.

    Parameters:
    - F: Thrust (N)
    - p_c: Combustion chamber pressure (Pa)
    - A_t: Nozzle throat area (m^2)

    Returns:
    - Thrust coefficient (dimensionless)
    """
    if A_t == 0:
        raise ZeroDivisionError("Throat area cannot be zero for thrust coefficient calculation.")
    return F / (p_c * A_t)

def total_impulse(F: float, burn_time: float) -> float:
    """
    Calculate the total impulse.

    Parameters:
    - F: Thrust (N)
    - burn_time: Burn time (s)

    Returns:
    - Total impulse (N·s)
    """
    return F * burn_time

def regenerative_cooling(Q_dot: float, m_c: float, C_p: float, T_in: float, T_out: float) -> float:
    """
    Calculate the rate of heat transfer in regenerative cooling.
    Note: Assumes constant properties of the coolant which might not hold in real scenarios.

    Parameters:
    - Q_dot: Rate of heat transfer (W)
    - m_c: Mass flow rate of coolant (kg/s)
    - C_p: Specific heat capacity of coolant (J/kg/K)
    - T_in: Inlet temperature of coolant (K)
    - T_out: Outlet temperature of coolant (K)

    Returns:
    - Heat transfer rate (W)
    """
    return m_c * C_p * (T_out - T_in)

def theoretical_specific_impulse(gamma: float, R: float, T_c: float, p_e: float, p_c: float, g_0: float = G_0) -> float:
    """
    Estimate theoretical specific impulse.

    Parameters:
    - gamma: Specific heat ratio (dimensionless)
    - R: Gas constant (J/kg/K)
    - T_c: Combustion chamber temperature (K)
    - p_e: Exit pressure (Pa)
    - p_c: Combustion chamber pressure (Pa)
    - g_0: Gravitational acceleration (m/s^2, default is from module constant)

    Returns:
    - Theoretical specific impulse (s)
    """
    term1 = (2 * gamma / (gamma - 1)) * R * T_c
    term2 = 1 - (p_e / p_c)**((gamma - 1) / gamma)
    return (1 / g_0) * (term1 * term2)**0.5  # Using sqrt for readability