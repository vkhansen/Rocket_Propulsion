import numpy as np

# Constants
G_0 = 9.81  # m/s^2, gravitational acceleration at Earth's surface
SIGMA_MATERIAL = 276e6  # Pa, yield strength of 6061-T6 Aluminum
FACTOR_OF_SAFETY = 1.5

# Vectorized Functions

def thrust(m_dot, v_e, p_e, p_0, A_e):
    """
    Vectorized thrust calculation
    """
    return m_dot * v_e + (p_e - p_0) * A_e

def exhaust_velocity(I_sp, g_0=G_0):
    """
    Vectorized exhaust velocity calculation
    """
    return I_sp * g_0

def delta_v(v_e, m_0, m_f):
    """
    Vectorized delta-v calculation
    """
    return v_e * np.log(m_0 / m_f)

def calculate_tank_dimensions(mass_propellant, rho_propellant):
    """
    Vectorized tank dimensions calculation
    """
    volume = mass_propellant / rho_propellant
    radius = (3 * volume / (4 * np.pi))**(1/3)
    return radius, volume

def calculate_wall_thickness(radius, pressure, sigma_material=SIGMA_MATERIAL, factor_of_safety=FACTOR_OF_SAFETY):
    """
    Vectorized wall thickness calculation
    """
    return (pressure * radius) / (sigma_material / factor_of_safety)

def estimate_safety_margin(actual_stress, allowable_stress):
    """
    Vectorized safety margin calculation
    """
    return allowable_stress / actual_stress

def calculate_dry_mass(mass_propellant, structural_mass_fraction):
    """
    Vectorized dry mass calculation
    """
    return mass_propellant / (1 - structural_mass_fraction)

# New Functions

def mass_flow_rate(mass_propellant, burn_time):
    """
    Vectorized mass flow rate calculation
    """
    return mass_propellant / burn_time

def calculate_pressure_ratio(p_e, p_0):
    """
    Vectorized pressure ratio calculation
    """
    return p_e / p_0

def specific_impulse(v_e, g_0=G_0):
    """
    Vectorized specific impulse calculation
    """
    return v_e / g_0

def mass_flow_rate_injector(p_inject, A_injector, rho_liquid, Cd):
    """
    Vectorized mass flow rate for liquid injector in hybrid motor
    """
    return Cd * A_injector * np.sqrt(2 * rho_liquid * p_inject)

def mass_flow_rate_nozzle(p_c, A_t, R, T, gamma):
    """
    Vectorized mass flow rate for nozzle with choked flow
    """
    return (p_c * A_t / np.sqrt(R * T)) * np.sqrt(gamma * (2 / (gamma + 1))**((gamma + 1) / (gamma - 1)))

# Main Script for Example Integration
mass_propellant = np.array([100, 200, 300])  # kg
rho_propellant = 820  # kg/m^3
pressure = 50e5  # Pa
structural_mass_fraction = 0.15

# Propulsion
v_e = exhaust_velocity(250)
delta_v_values = delta_v(v_e, np.array([400, 800, 1200]), mass_propellant)

# Tank Dimensions
tank_radius, tank_volume = calculate_tank_dimensions(mass_propellant, rho_propellant)
wall_thickness = calculate_wall_thickness(tank_radius, pressure)

# Safety Margin
actual_stress = pressure * tank_radius / wall_thickness
safety_margin = estimate_safety_margin(actual_stress, SIGMA_MATERIAL / FACTOR_OF_SAFETY)

# Dry Mass
dry_mass = calculate_dry_mass(mass_propellant, structural_mass_fraction)

# Display Results
print(f'Tank Radius (m): {tank_radius}')
print(f'Tank Volume (m^3): {tank_volume}')
print(f'Wall Thickness (m): {wall_thickness}')
print(f'Safety Margin: {safety_margin}')
print(f'Delta-V (m/s): {delta_v_values}')
print(f'Dry Mass (kg): {dry_mass}')