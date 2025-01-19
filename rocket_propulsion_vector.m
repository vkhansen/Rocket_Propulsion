% Constants
G_0 = 9.81; % m/s^2, gravitational acceleration at Earth's surface
SIGMA_MATERIAL = 276e6; % Pa, yield strength of 6061-T6 Aluminum
FACTOR_OF_SAFETY = 1.5;

% Vectorized Functions

% Thrust calculation
function thrust_values = thrust(m_dot, v_e, p_e, p_0, A_e)
    % Vectorized thrust calculation
    thrust_values = m_dot .* v_e + (p_e - p_0) .* A_e;
end

% Exhaust velocity calculation
function v_e = exhaust_velocity(I_sp, g_0)
    if nargin < 2
        g_0 = G_0;
    end
    v_e = I_sp * g_0;
end

% Delta-v calculation
function delta_v_values = delta_v(v_e, m_0, m_f)
    delta_v_values = v_e .* log(m_0 ./ m_f);
end

% Tank dimensions calculation
function [radius, volume] = calculate_tank_dimensions(mass_propellant, rho_propellant)
    volume = mass_propellant ./ rho_propellant;
    radius = (3 * volume / (4 * pi)).^(1/3);
end

% Wall thickness calculation
function wall_thickness = calculate_wall_thickness(radius, pressure, sigma_material, factor_of_safety)
    if nargin < 3
        sigma_material = SIGMA_MATERIAL;
    end
    if nargin < 4
        factor_of_safety = FACTOR_OF_SAFETY;
    end
    wall_thickness = (pressure .* radius) ./ (sigma_material / factor_of_safety);
end

% Safety margin calculation
function safety_margin = estimate_safety_margin(actual_stress, allowable_stress)
    safety_margin = allowable_stress ./ actual_stress;
end

% Dry mass calculation
function dry_mass = calculate_dry_mass(mass_propellant, structural_mass_fraction)
    dry_mass = mass_propellant ./ (1 - structural_mass_fraction);
end

% Mass flow rate calculation
function m_dot = mass_flow_rate(mass_propellant, burn_time)
    % Vectorized mass flow rate calculation
    m_dot = mass_propellant ./ burn_time;
end

% Pressure ratio calculation
function pressure_ratio = calculate_pressure_ratio(p_e, p_0)
    % Vectorized pressure ratio calculation
    pressure_ratio = p_e ./ p_0;
end

% Specific Impulse calculation
function I_sp = specific_impulse(v_e, g_0)
    if nargin < 2
        g_0 = G_0;
    end
    I_sp = v_e ./ g_0;
end

% Mass flow rate for liquid injector in hybrid motor
function m_dot_injector = mass_flow_rate_injector(p_inject, A_injector, rho_liquid, Cd)
    % Vectorized mass flow rate for liquid injector in hybrid motor
    m_dot_injector = Cd .* A_injector .* sqrt(2 .* rho_liquid .* p_inject);
end

% Nozzle mass flow rate with choked flow
function m_dot_nozzle = mass_flow_rate_nozzle(p_c, A_t, R, T, gamma)
    % Vectorized mass flow rate for nozzle with choked flow
    m_dot_nozzle = (p_c .* A_t ./ sqrt(R .* T)) .* sqrt(gamma .* (2 ./ (gamma + 1)).^((gamma + 1) ./ (gamma - 1)));
end

% Main Script for Example Integration
mass_propellant = [100, 200, 300]; % kg
rho_propellant = 820; % kg/m^3
pressure = 50e5; % Pa
structural_mass_fraction = 0.15;

% Example parameters for new functions
burn_time = [10, 20, 30]; % seconds
p_inject = 1e6; % Pa, injection pressure for hybrid motor
A_injector = 0.01; % m^2, injector area
rho_liquid = 1000; % kg/m^3, liquid density
Cd = 0.95; % Discharge coefficient for injector
p_c = 1e6; % Pa, chamber pressure for nozzle calculation
A_t = 0.005; % m^2, throat area
R = 287; % J/kg*K, gas constant for air
T = 300; % K, temperature
gamma = 1.4; % Specific heat ratio for air

% Propulsion
v_e = exhaust_velocity(250);
delta_v_values = delta_v(v_e, [400, 800, 1200], mass_propellant);

% Tank Dimensions
[tank_radius, tank_volume] = calculate_tank_dimensions(mass_propellant, rho_propellant);
wall_thickness = calculate_wall_thickness(tank_radius, pressure);

% Safety Margin
actual_stress = pressure .* tank_radius ./ wall_thickness;
safety_margin = estimate_safety_margin(actual_stress, SIGMA_MATERIAL / FACTOR_OF_SAFETY);

% Dry Mass
dry_mass = calculate_dry_mass(mass_propellant, structural_mass_fraction);

% Additional Calculations
m_dot = mass_flow_rate(mass_propellant, burn_time);
m_dot_injector = mass_flow_rate_injector(p_inject, A_injector, rho_liquid, Cd);
m_dot_nozzle = mass_flow_rate_nozzle(p_c, A_t, R, T, gamma);
pressure_ratio = calculate_pressure_ratio(pressure, 101325); % Assuming atmospheric pressure at sea level for p_0
I_sp = specific_impulse(v_e);

% Display Results
fprintf('Tank Radius (m): %s\n', mat2str(tank_radius));
fprintf('Tank Volume (m^3): %s\n', mat2str(tank