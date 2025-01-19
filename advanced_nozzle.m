% Constants (not explicitly defined in the document, but used in functions)
G_0 = 9.81;  % m/s^2, gravitational acceleration at Earth's surface

% Thrust Calculation
function F = thrust(m_dot, v_e, p_e, p_amb, A_e)
    % Calculate the total thrust of a rocket engine
    %
    % Parameters:
    %   m_dot - Mass flow rate (kg/s)
    %   v_e   - Effective exhaust velocity (m/s)
    %   p_e   - Pressure at nozzle exit (Pa)
    %   p_amb - Ambient pressure (Pa)
    %   A_e   - Area of nozzle exit (m^2)
    %
    % Returns:
    %   F - Thrust (N)
    F = m_dot * v_e + (p_e - p_amb) * A_e;
end

% Delta-V Calculation (Tsiolkovsky rocket equation)
function delta_v = delta_v(v_e, m_0, m_f)
    % Calculate the delta-v using the Tsiolkovsky rocket equation
    %
    % Parameters:
    %   v_e - Effective exhaust velocity (m/s)
    %   m_0 - Initial mass of the rocket (kg)
    %   m_f - Final mass of the rocket (kg)
    %
    % Returns:
    %   delta_v - Change in velocity (m/s)
    delta_v = v_e * log(m_0 / m_f);
end

% Specific Impulse Calculation
function I_sp = specific_impulse(F, m_dot)
    % Calculate the specific impulse
    %
    % Parameters:
    %   F     - Thrust (N)
    %   m_dot - Mass flow rate (kg/s)
    %
    % Returns:
    %   I_sp - Specific impulse (s)
    I_sp = F / (m_dot * G_0);
end

% Mass Flow Rate Calculation
function m_dot = mass_flow_rate(rho_e, A_e, v_e)
    % Calculate the mass flow rate
    %
    % Parameters:
    %   rho_e - Density of exhaust gases at nozzle exit (kg/m^3)
    %   A_e   - Area of nozzle exit (m^2)
    %   v_e   - Velocity of exhaust gases (m/s)
    %
    % Returns:
    %   m_dot - Mass flow rate (kg/s)
    m_dot = rho_e * A_e * v_e;
end

% Nozzle Area Ratio Calculation
function area_ratio = nozzle_area_ratio(A_e, A_t, gamma, M_e)
    % Calculate the nozzle area ratio
    %
    % Parameters:
    %   A_e   - Area of nozzle exit (m^2)
    %   A_t   - Area of nozzle throat (m^2)
    %   gamma - Ratio of specific heats
    %   M_e   - Mach number at nozzle exit
    %
    % Returns:
    %   area_ratio - Nozzle area ratio
    term1 = (gamma + 1) / 2;
    term2 = 1 + (gamma - 1) / 2 * M_e^2;
    area_ratio = (A_e / A_t) / ((term1^(term1 / (gamma - 1))) * (1 / M_e) * (term2^((gamma + 1) / (2 * (gamma - 1)))));
end

% Thermal Efficiency Calculation (simplified Carnot efficiency)
function efficiency = thermal_efficiency(T_e, T_c)
    % Calculate the thermal efficiency
    %
    % Parameters:
    %   T_e - Temperature at nozzle exit (K)
    %   T_c - Temperature in combustion chamber (K)
    %
    % Returns:
    %   efficiency - Thermal efficiency (fraction)
    efficiency = 1 - (T_e / T_c);
end

% Characteristic Velocity Calculation
function c_star = characteristic_velocity(p_c, A_t, m_dot)
    % Calculate the characteristic velocity (c*)
    %
    % Parameters:
    %   p_c   - Chamber pressure (Pa)
    %   A_t   - Area of nozzle throat (m^2)
    %   m_dot - Mass flow rate (kg/s)
    %
    % Returns:
    %   c_star - Characteristic velocity (m/s)
    c_star = (p_c * A_t) / m_dot;
end

% Thrust Coefficient Calculation
function C_F = thrust_coefficient(F, p_c, A_t)
    % Calculate the thrust coefficient
    %
    % Parameters:
    %   F   - Thrust (N)
    %   p_c - Chamber pressure (Pa)
    %   A_t - Area of nozzle throat (m^2)
    %
    % Returns:
    %   C_F - Thrust coefficient
    C_F = F / (p_c * A_t);
end

% Total Impulse Calculation
function total_impulse = total_impulse(F, burn_time)
    % Calculate the total impulse
    %
    % Parameters:
    %   F - Thrust (N)
    %   burn_time - Duration of burn (s)
    %
    % Returns:
    %   total_impulse - Total impulse (Ns)
    total_impulse = F * burn_time;
end