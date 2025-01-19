function main()
    % Constants
    G_0 = 9.81; % m/s^2, gravitational acceleration at Earth's surface

    % Example calls to functions (commented out for demonstration)
    % thrust_value = thrust(10, 3000, 101325, 101325, 0.05);
    % exhaust_vel = exhaust_velocity(300);
    % delta_v_value = delta_v(3000, 1000, 100);
    % specific_impulse_value = specific_impulse(10000, 10);
    % mass_flow = mass_flow_rate(1.2, 0.05, 3000);
    % area_ratio = nozzle_area_ratio(0.1, 0.01, 1.4, 2);
    % thermal_eff = thermal_efficiency(1000, 3000);
    % c_star = characteristic_velocity(1e6, 0.01, 10);
    % C_F = thrust_coefficient(1e5, 1e6, 0.01);
    % total_impulse_value = total_impulse(10000, 100);
    % heat_transfer = regenerative_cooling(1e6, 5, 500, 300, 400);
    % I_sp_theoretical = theoretical_specific_impulse(1.4, 287, 3000, 101325, 6e6);
end

function F = thrust(m_dot, v_e, p_e, p_0, A_e)
    % Calculate the total thrust of a rocket engine.
    if p_e < p_0
        error('Exit pressure should not be less than ambient pressure for positive thrust contribution.');
    end
    F = m_dot * v_e + (p_e - p_0) * A_e;
end

function v_e = exhaust_velocity(I_sp, g_0)
    % Calculate the effective exhaust velocity.
    if nargin < 2, g_0 = 9.81; end
    v_e = I_sp * g_0;
end

function delta_v = delta_v(v_e, m_0, m_f)
    % Calculate the delta-v using the Tsiolkovsky rocket equation.
    if m_0 <= m_f
        error('Initial mass must be greater than final mass for positive delta-v.');
    end
    delta_v = v_e * log(m_0 / m_f);
end

function I_sp = specific_impulse(F, m_dot, g_0)
    % Calculate the specific impulse.
    if nargin < 3, g_0 = 9.81; end
    if m_dot == 0
        error('Mass flow rate cannot be zero for specific impulse calculation.');
    end
    I_sp = F / (m_dot * g_0);
end

function m_dot = mass_flow_rate(rho_e, A_e, v_e)
    % Calculate the mass flow rate.
    m_dot = rho_e * A_e * v_e;
end

function area_ratio = nozzle_area_ratio(A_e, A_t, gamma, M_e)
    % Calculate the nozzle area ratio.
    term1 = (gamma + 1) / 2;
    term2 = 1 + (gamma - 1) / 2 * M_e^2;
    area_ratio = (A_e / A_t) / ((term1^(term1 / (gamma - 1))) * (1 / M_e) * (term2^((gamma + 1) / (2 * (gamma - 1)))));
end

function efficiency = thermal_efficiency(T_e, T_c)
    % Calculate the thermal efficiency (simplified Carnot efficiency).
    if T_e >= T_c
        error('Exhaust temperature cannot be greater than or equal to combustion chamber temperature.');
    end
    efficiency = 1 - (T_e / T_c);
end

function c_star = characteristic_velocity(p_c, A_t, m_dot)
    % Calculate the characteristic velocity (c*).
    if m_dot == 0
        error('Mass flow rate cannot be zero for characteristic velocity calculation.');
    end
    c_star = (p_c * A_t) / m_dot;
end

function C_F = thrust_coefficient(F, p_c, A_t)
    % Calculate the thrust coefficient.
    if A_t == 0
        error('Throat area cannot be zero for thrust coefficient calculation.');
    end
    C_F = F / (p_c * A_t);
end

function total_impulse = total_impulse(F, burn_time)
    % Calculate the total impulse.
    total_impulse = F * burn_time;
end

function Q_dot = regenerative_cooling(Q_dot, m_c, C_p, T_in, T_out)
    % Calculate the rate of heat transfer in regenerative cooling.
    Q_dot = m_c * C_p * (T_out - T_in);
end

function I_sp = theoretical_specific_impulse(gamma, R, T_c, p_e, p_c, g_0)
    % Estimate theoretical specific impulse.
    if nargin < 6, g_0 = 9.81; end
    term1 = (2 * gamma / (gamma - 1)) * R * T_c;
    term2 = 1 - (p_e / p_c)^((gamma - 1) / gamma);
    I_sp = (1 / g_0) * sqrt(term1 * term2);
end