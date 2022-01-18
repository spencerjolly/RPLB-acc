import numpy as np
from numba import jit

@jit(nopython=True)
def RPLB_acc_anySTC_2D(lambda_0, tau_0, w_00, P, Psi_0, spec_phase_coeffs, z_0, r_0, beta_0, LC_coeffs, g_0):
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
    r = np.empty(shape=(len(time)))
    theta = np.empty(shape=(len(time)))
    v_z = np.empty(shape=(len(time)))
    v_r = np.empty(shape=(len(time)))
    v_t = np.empty(shape=(len(time)))
    gamma = np.empty(shape=(len(time)))
    deriv2 = np.empty(shape=(len(time)))
    deriv4 = np.empty(shape=(len(time)))

    # Set initial conditions
    z[0] = beta_0*c*time[0] + z_0
    r[0] = r_0
    theta[0] = 0.0
    v_z[0] = beta_0*c
    v_r[0] = 0.0
    v_t[0] = 0.0
    gamma[0] = 1/np.sqrt(1-beta_0**2)

    # do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):

        phi_G = np.arctan((z[k]-z_omega)/z_R)
        w = w_0*np.sqrt(1+((z[k]-z_omega)/z_R)**2)
        R_inv = (z[k]-z_omega)/((z[k]-z_omega)**2 + z_R**2)
        phi_norm = Psi_0-(omega/c)*(z[k]+(R_inv*(x[k]-x_omega)**2)/2)+omega*time[k]
        trans = np.exp(-((x[k]-x_omega)/w)**2)

        c_2 = (w_0/w)**2 * np.exp(1j*(phi_norm + 2*phi_G))
        c_3 = (w_0/w)**3 * np.exp(1j*(phi_norm + 3*phi_G))
        c_4 = (w_0/w)**4 * np.exp(1j*(phi_norm + 4*phi_G))
        c_5 = (w_0/w)**5 * np.exp(1j*(phi_norm + 5*phi_G))
        c_6 = (w_0/w)**6 * np.exp(1j*(phi_norm + 6*phi_G))
        c_7 = (w_0/w)**7 * np.exp(1j*(phi_norm + 7*phi_G))
        c_8 = (w_0/w)**8 * np.exp(1j*(phi_norm + 8*phi_G))

        rho = r[k]/w_0

        B_z_spec = -1*pulse_prep*((c_2 - c_3*rho**2)*eps**2 +
                               ((1/2)*c_3 + (1/2)*c_4*rho**2 - (5/4)*c_5*rho**4 + (1/4)*c_6*rho**6)*eps**4)/c
        B_z_time = np.sum(B_z_spec*trans)*omega_step/(delta_omega*np.sqrt(np.pi))

        E_t_spec = pulse_prep*((c_2*rho)*eps +
                               (-(1/2)*c_3*rho + c_4*rho**3 - (1/4)*c_5*rho**5)*eps**3 +
                               (-(3/8)*c_4*rho - (3/8)*c_5*rho**3 + (17/16)*c_6*rho**5 -
                                (3/8)*c_7*rho**7 + (1/32)*c_8*rho**9)*eps**5)*np.exp(+1j*np.pi/2)
        E_t_time = np.sum(E_t_spec*trans)*omega_step/(delta_omega*np.sqrt(np.pi))

        B_r_spec = -1*pulse_prep*((c_2*rho)*eps +
                               ((1/2)*c_3*rho + (1/2)*c_4*rho**3 - (1/4)*c_5*rho**5)*eps**3 +
                               ((3/8)*c_4*rho + (3/8)*c_5*rho**3 + (3/16)*c_6*rho**5 -
                                (1/4)*c_7*rho**7 + (1/32)*c_8*rho**9)*eps**5)*np.exp(+1j*np.pi/2)/c
        B_r_time = np.sum(B_r_spec*trans)*omega_step/(delta_omega*np.sqrt(np.pi))

        B_z_total = np.real(Amp*B_z_time)
        E_t_total = np.real(Amp*E_t_time)
        dot_product = v_t[k]*E_t_total
        B_r_total = np.real(Amp*B_r_time)

        deriv2[k] = (-q_e/(gamma[k]*m_e))*(XXX-v_z[k]*dot_product/(c**2))
        deriv4[k] = (-q_e/(gamma[k]*m_e))*(XXX-v_r[k]*dot_product/(c**2))
        deriv6[k] = (-q_e/(gamma[k]*m_e))*(E_t_total+XXX-v_t[k]*dot_product/(c**2))

        if k==0:
            z[k+1] = z[k] + dt*v_z[k]
            v_z[k+1] = v_z[k] + dt*deriv2[k]
            r[k+1] = r[k] + dt*v_r[k]
            v_r[k+1] = v_r[k] + dt*deriv4[k]
            theta[k+1] = theta[k] + dt*v_t[k]
            v_t[k+1] = v_t[k] + dt*deriv6[k]
        elif k==1:
            z[k+1] = z[k] + dt*(1.5*v_z[k]-0.5*v_z[k-1])
            v_z[k+1] = v_z[k] + dt*(1.5*deriv2[k]-0.5*deriv2[k-1])
            r[k+1] = r[k] + dt*(1.5*v_r[k]-0.5*v_r[k-1])
            v_r[k+1] = v_r[k] + dt*(1.5*deriv4[k]-0.5*deriv4[k-1])
            theta[k+1] = theta[k] + dt*(1.5*v_t[k]-0.5*v_t[k-1])
            v_t[k+1] = v_t[k] + dt*(1.5*deriv6[k]-0.5*deriv6[k-1])
        elif k==2:
            z[k+1] = z[k] + dt*((23/12)*v_z[k]-(4/3)*v_z[k-1]+(5/12)*v_z[k-2])
            v_z[k+1] = v_z[k] + dt*((23/12)*deriv2[k]-(4/3)*deriv2[k-1]+(5/12)*deriv2[k-2])
            r[k+1] = r[k] + dt*((23/12)*v_r[k]-(4/3)*v_r[k-1]+(5/12)*v_r[k-2])
            v_r[k+1] = v_r[k] + dt*((23/12)*deriv4[k]-(4/3)*deriv4[k-1]+(5/12)*deriv4[k-2])
            theta[k+1] = theta[k] + dt*((23/12)*v_t[k]-(4/3)*v_t[k-1]+(5/12)*v_t[k-2])
            v_t[k+1] = v_t[k] + dt*((23/12)*deriv6[k]-(4/3)*deriv6[k-1]+(5/12)*deriv6[k-2])
        elif k==3:
            z[k+1] = z[k] + dt*((55/24)*v_z[k]-(59/24)*v_z[k-1]+(37/24)*v_z[k-2]-(3/8)*v_z[k-3])
            v_z[k+1] = v_z[k] + dt*((55/24)*deriv2[k]-(59/24)*deriv2[k-1]+(37/24)*deriv2[k-2]-(3/8)*deriv2[k-3])
            r[k+1] = r[k] + dt*((55/24)*v_r[k]-(59/24)*v_r[k-1]+(37/24)*v_r[k-2]-(3/8)*v_r[k-3])
            v_r[k+1] = v_r[k] + dt*((55/24)*deriv4[k]-(59/24)*deriv4[k-1]+(37/24)*deriv4[k-2]-(3/8)*deriv4[k-3])
            theta[k+1] = theta[k] + dt*((55/24)*v_t[k]-(59/24)*v_t[k-1]+(37/24)*v_t[k-2]-(3/8)*v_t[k-3])
            v_t[k+1] = v_t[k] + dt*((55/24)*deriv6[k]-(59/24)*deriv6[k-1]+(37/24)*deriv6[k-2]-(3/8)*deriv6[k-3])
        else:
            z[k+1] = z[k] + dt*((1901/720)*v_z[k]-(1387/360)*v_z[k-1]+(109/30)*v_z[k-2]-(637/360)*v_z[k-3]+(251/720)*v_z[k-4])
            v_z[k+1] = v_z[k] + dt*((1901/720)*deriv2[k]-(1387/360)*deriv2[k-1]+(109/30)*deriv2[k-2]-(637/360)*deriv2[k-3]+(251/720)*deriv2[k-4])
            r[k+1] = r[k] + dt*((1901/720)*v_r[k]-(1387/360)*v_r[k-1]+(109/30)*v_r[k-2]-(637/360)*v_r[k-3]+(251/720)*v_r[k-4])
            v_r[k+1] = v_r[k] + dt*((1901/720)*deriv4[k]-(1387/360)*deriv4[k-1]+(109/30)*deriv4[k-2]-(637/360)*deriv4[k-3]+(251/720)*deriv4[k-4])
            theta[k+1] = theta[k] + dt*((1901/720)*v_t[k]-(1387/360)*v_t[k-1]+(109/30)*v_t[k-2]-(637/360)*v_t[k-3]+(251/720)*v_t[k-4])
            v_t[k+1] = v_t[k] + dt*((1901/720)*deriv6[k]-(1387/360)*deriv6[k-1]+(109/30)*deriv6[k-2]-(637/360)*deriv6[k-3]+(251/720)*deriv6[k-4])

        gamma[k+1] = 1/np.sqrt(1-(v_z[k+1]**2+v_r[k+1]**2+v_t[k+1]**2)/c**2)

    KE = (gamma-1)*m_e*c**2/q_e
    return time, z, r, theta, v_z, v_r, v_t, KE[-1]