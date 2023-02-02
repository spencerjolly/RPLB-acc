import numpy as np
from numba import jit

@jit(nopython=True)
def APLB_acc_anySTC_3D(lambda_0, tau_0, w_00, P, Psi_0, spec_phase_coeffs, LC_coeffs, SC_coeffs, g_0, z_0, x_0, y_0, beta_0):
    # initialize constants (SI units)
    c = 2.99792458e8 #speed of light
    m_e = 9.10938356e-31
    q_e = 1.60217662e-19
    e_0 = 8.85418782e-12
    # calculate frequency properties
    omega_0 = 2*np.pi*c/lambda_0
    delta_omega = 2/tau_0
    # calculate Rayleigh range
    z_R0 = (omega_0*w_00**2)/(2*c)
    
    t_start = -50*tau_0
    t_end = 1400*tau_0
    n = 200  # number of time steps per laser period
    num_t = np.int_(np.round(n*(t_end-t_start)/(lambda_0/c)))
    time = np.linspace(t_start, t_end, num_t)
    dt = time[1]-time[0]

    omega = np.linspace((omega_0-4*delta_omega), (omega_0+4*delta_omega), 300)
    omega_step = omega[1]-omega[0]

    pulse_temp = np.exp(-((omega-omega_0)/delta_omega)**2)  # spectral envelope
    
    # Spectral phase components
    spec_phase = np.zeros(shape=len(omega))
    temp = 1
    for i in range(0, len(spec_phase_coeffs)):
        temp = temp*(i+2)
        spec_phase = spec_phase+(spec_phase_coeffs[i]/temp)*(omega-omega_0)**(i+2)
    
    pulse_prep = pulse_temp*np.exp(-1j*spec_phase)  # adding spectral phase to envelope
    
    # Frequency dependent longitudinal waist position due to chromaticity
    z_omega = np.zeros(shape=len(omega))
    temp = 1
    for j in range(0, len(LC_coeffs)):
        temp = temp*(j+1)
        z_omega = z_omega+z_R0*(LC_coeffs[j]/temp)*(omega-omega_0)**(j+1)
        
    # Frequency dependent transverse focal position due to spatial chirp
    x_omega = np.zeros(shape=len(omega))
    temp = 1
    for j in range(0, len(LC_coeffs)):
        temp = temp*(j+1)
        x_omega = x_omega+(w_00/2)*(SC_coeffs[j]/temp)*(omega-omega_0)**(j+1)
    
    # frequency dependent beam parameters based on Porras factor g_0
    w_0 = w_00*(omega_0/omega)**((g_0+1)/2)
    z_R = z_R0*(omega_0/omega)**(g_0)

    # perturbation parameter
    eps = w_0/z_R
    # amplitude factor
    P_corr = 1 + 3*(eps/2)**2 + 9*(eps/2)**4
    Amp = np.sqrt(8*P/(P_corr*np.pi*e_0*c)) * (omega/(2*c))

    # initialize empty arrays
    z = np.empty(shape=(len(time)))
    x = np.empty(shape=(len(time)))
    y = np.empty(shape=(len(time)))
    v_z = np.empty(shape=(len(time)))
    v_x = np.empty(shape=(len(time)))
    v_y = np.empty(shape=(len(time)))
    gamma = np.empty(shape=(len(time)))
    KE = np.zeros(shape=(len(time)))
    deriv2 = np.empty(shape=(len(time)))
    deriv4 = np.empty(shape=(len(time)))
    deriv6 = np.empty(shape=(len(time)))

    # Set initial conditions
    z[0] = beta_0*c*time[0] + z_0
    x[0] = x_0
    y[0] = y_0
    v_z[0] = beta_0*c
    v_x[0] = 0.0
    v_y[0] = 0.0
    gamma[0] = 1/np.sqrt(1-beta_0**2)
    KE[0] = ((1/np.sqrt(1-beta_0**2))-1)*m_e*c**2/q_e
    k_stop = -1

    # do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):

        rho = np.sqrt((x[k]-x_omega)**2 + y[k]**2)/w_0
        
        phi_G = np.arctan((z[k]-z_omega)/z_R)
        w = w_0*np.sqrt(1+((z[k]-z_omega)/z_R)**2)
        R_inv = (z[k]-z_omega)/((z[k]-z_omega)**2 + z_R**2)
        phi_norm = Psi_0-(omega/c)*(z[k]+(R_inv*(rho*w_0)**2)/2)+omega*time[k]
        trans = np.exp(-(rho*w_0/w)**2)

        c_2 = (w_0/w)**2 * np.exp(1j*(phi_norm + 2*phi_G))
        c_3 = (w_0/w)**3 * np.exp(1j*(phi_norm + 3*phi_G))
        c_4 = (w_0/w)**4 * np.exp(1j*(phi_norm + 4*phi_G))
        c_5 = (w_0/w)**5 * np.exp(1j*(phi_norm + 5*phi_G))
        c_6 = (w_0/w)**6 * np.exp(1j*(phi_norm + 6*phi_G))
        c_7 = (w_0/w)**7 * np.exp(1j*(phi_norm + 7*phi_G))
        c_8 = (w_0/w)**8 * np.exp(1j*(phi_norm + 8*phi_G))

        B_z_spec = -1*pulse_prep*((c_2 - c_3*rho**2)*eps**2 +
                               ((1/2)*c_3 + (1/2)*c_4*rho**2 - (5/4)*c_5*rho**4 + (1/4)*c_6*rho**6)*eps**4)/c
        B_z_time = np.sum(Amp*B_z_spec*trans)*omega_step/(delta_omega*np.sqrt(np.pi))

        B_x_spec = -1*pulse_prep*((x[k]-x_omega)/w_0)*((c_2)*eps +
                               (-(1/2)*c_3 + c_4*rho**2 - (1/4)*c_5*rho**4)*eps**3 +
                               (-(3/8)*c_4 - (3/8)*c_5*rho**2 + (17/16)*c_6*rho**4 -
                                (3/8)*c_7*rho**6 + (1/32)*c_8*rho**8)*eps**5)*np.exp(+1j*np.pi/2)/c
        B_x_time = np.sum(Amp*B_x_spec*trans)*omega_step/(delta_omega*np.sqrt(np.pi))
        
        B_y_spec = -1*pulse_prep*(y[k]/w_0)*((c_2)*eps +
                               (-(1/2)*c_3 + c_4*rho**2 - (1/4)*c_5*rho**4)*eps**3 +
                               (-(3/8)*c_4 - (3/8)*c_5*rho**2 + (17/16)*c_6*rho**4 -
                                (3/8)*c_7*rho**6 + (1/32)*c_8*rho**8)*eps**5)*np.exp(+1j*np.pi/2)/c
        B_y_time = np.sum(Amp*B_y_spec*trans)*omega_step/(delta_omega*np.sqrt(np.pi))

        E_x_spec = pulse_prep*(-y[k]/w_0)*((c_2)*eps +
                               ((1/2)*c_3 + (1/2)*c_4*rho**2 - (1/4)*c_5*rho**4)*eps**3 +
                               ((3/8)*c_4 + (3/8)*c_5*rho**2 + (3/16)*c_6*rho**4 -
                                (1/4)*c_7*rho**6 + (1/32)*c_8*rho**8)*eps**5)*np.exp(+1j*np.pi/2)
        E_x_time = np.sum(Amp*E_x_spec*trans)*omega_step/(delta_omega*np.sqrt(np.pi))
        
        E_y_spec = pulse_prep*((x[k]-x_omega)/w_0)*((c_2)*eps +
                               ((1/2)*c_3 + (1/2)*c_4*rho**2 - (1/4)*c_5*rho**4)*eps**3 +
                               ((3/8)*c_4 + (3/8)*c_5*rho**2 + (3/16)*c_6*rho**4 -
                                (1/4)*c_7*rho**6 + (1/32)*c_8*rho**8)*eps**5)*np.exp(+1j*np.pi/2)
        E_y_time = np.sum(Amp*E_y_spec*trans)*omega_step/(delta_omega*np.sqrt(np.pi))

        B_z_total = np.real(B_z_time)
        E_x_total = np.real(E_x_time)
        E_y_total = np.real(E_y_time)
        dot_product = v_x[k]*E_x_total + v_y[k]*E_y_total
        B_x_total = np.real(B_x_time)
        B_y_total = np.real(B_y_time)

        deriv2[k] = (-q_e/(gamma[k]*m_e))*(v_x[k]*B_y_total-v_y[k]*B_x_total-v_z[k]*dot_product/(c**2))  # Force in z
        deriv4[k] = (-q_e/(gamma[k]*m_e))*(E_x_total+v_y[k]*B_z_total-v_z[k]*B_y_total-v_x[k]*dot_product/(c**2))  # Force in x
        deriv6[k] = (-q_e/(gamma[k]*m_e))*(E_y_total+v_z[k]*B_x_total-v_x[k]*B_z_total-v_y[k]*dot_product/(c**2))  # Force in y

        if k==0:
            z[k+1] = z[k] + dt*v_z[k]
            v_z[k+1] = v_z[k] + dt*deriv2[k]
            x[k+1] = x[k] + dt*v_x[k]
            v_x[k+1] = v_x[k] + dt*deriv4[k]
            y[k+1] = y[k] + dt*v_y[k]
            v_y[k+1] = v_y[k] + dt*deriv6[k]
        elif k==1:
            z[k+1] = z[k] + dt*(1.5*v_z[k]-0.5*v_z[k-1])
            v_z[k+1] = v_z[k] + dt*(1.5*deriv2[k]-0.5*deriv2[k-1])
            x[k+1] = x[k] + dt*(1.5*v_x[k]-0.5*v_x[k-1])
            v_x[k+1] = v_x[k] + dt*(1.5*deriv4[k]-0.5*deriv4[k-1])
            y[k+1] = y[k] + dt*(1.5*v_y[k]-0.5*v_y[k-1])
            v_y[k+1] = v_y[k] + dt*(1.5*deriv6[k]-0.5*deriv6[k-1])
        elif k==2:
            z[k+1] = z[k] + dt*((23/12)*v_z[k]-(4/3)*v_z[k-1]+(5/12)*v_z[k-2])
            v_z[k+1] = v_z[k] + dt*((23/12)*deriv2[k]-(4/3)*deriv2[k-1]+(5/12)*deriv2[k-2])
            x[k+1] = x[k] + dt*((23/12)*v_x[k]-(4/3)*v_x[k-1]+(5/12)*v_x[k-2])
            v_x[k+1] = v_x[k] + dt*((23/12)*deriv4[k]-(4/3)*deriv4[k-1]+(5/12)*deriv4[k-2])
            y[k+1] = y[k] + dt*((23/12)*v_y[k]-(4/3)*v_y[k-1]+(5/12)*v_y[k-2])
            v_y[k+1] = v_y[k] + dt*((23/12)*deriv6[k]-(4/3)*deriv6[k-1]+(5/12)*deriv6[k-2])
        elif k==3:
            z[k+1] = z[k] + dt*((55/24)*v_z[k]-(59/24)*v_z[k-1]+(37/24)*v_z[k-2]-(3/8)*v_z[k-3])
            v_z[k+1] = v_z[k] + dt*((55/24)*deriv2[k]-(59/24)*deriv2[k-1]+(37/24)*deriv2[k-2]-(3/8)*deriv2[k-3])
            x[k+1] = x[k] + dt*((55/24)*v_x[k]-(59/24)*v_x[k-1]+(37/24)*v_x[k-2]-(3/8)*v_x[k-3])
            v_x[k+1] = v_x[k] + dt*((55/24)*deriv4[k]-(59/24)*deriv4[k-1]+(37/24)*deriv4[k-2]-(3/8)*deriv4[k-3])
            y[k+1] = y[k] + dt*((55/24)*v_y[k]-(59/24)*v_y[k-1]+(37/24)*v_y[k-2]-(3/8)*v_y[k-3])
            v_y[k+1] = v_y[k] + dt*((55/24)*deriv6[k]-(59/24)*deriv6[k-1]+(37/24)*deriv6[k-2]-(3/8)*deriv6[k-3])
        else:
            z[k+1] = z[k] + dt*((1901/720)*v_z[k]-(1387/360)*v_z[k-1]+(109/30)*v_z[k-2]-(637/360)*v_z[k-3]+(251/720)*v_z[k-4])
            v_z[k+1] = v_z[k] + dt*((1901/720)*deriv2[k]-(1387/360)*deriv2[k-1]+(109/30)*deriv2[k-2]-(637/360)*deriv2[k-3]+(251/720)*deriv2[k-4])
            x[k+1] = x[k] + dt*((1901/720)*v_x[k]-(1387/360)*v_x[k-1]+(109/30)*v_x[k-2]-(637/360)*v_x[k-3]+(251/720)*v_x[k-4])
            v_x[k+1] = v_x[k] + dt*((1901/720)*deriv4[k]-(1387/360)*deriv4[k-1]+(109/30)*deriv4[k-2]-(637/360)*deriv4[k-3]+(251/720)*deriv4[k-4])
            y[k+1] = y[k] + dt*((1901/720)*v_y[k]-(1387/360)*v_y[k-1]+(109/30)*v_y[k-2]-(637/360)*v_y[k-3]+(251/720)*v_y[k-4])
            v_y[k+1] = v_y[k] + dt*((1901/720)*deriv6[k]-(1387/360)*deriv6[k-1]+(109/30)*deriv6[k-2]-(637/360)*deriv6[k-3]+(251/720)*deriv6[k-4])

        gamma[k+1] = 1/np.sqrt(1-(v_z[k+1]**2+v_x[k+1]**2+v_y[k+1]**2)/c**2)
        KE[k+1] = (gamma[k+1]-1)*m_e*c**2/q_e

        if (time[k] > 300*tau_0 and np.mean(np.abs(np.diff(KE[k-np.int(10*n):k+1]))/(KE[k+1]*dt)) < 1e7):
            k_stop = k+1
            break

    return time[:k_stop], z[:k_stop], x[:k_stop], y[:k_stop], v_z[:k_stop], v_x[:k_stop], v_y[:k_stop], KE[:k_stop]