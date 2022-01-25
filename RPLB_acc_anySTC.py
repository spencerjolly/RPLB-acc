import numpy as np
from numba import jit

@jit(nopython=True)
def RPLB_acc_anySTC(lambda_0, tau_0, w_00, P, Psi_0, spec_phase_coeffs, LC_coeffs, g_0_coeffs, t_0, z_0, beta_0):
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
    # amplitude factor
    Amp = np.sqrt(8*P/(np.pi*e_0*c))
    # stretched pulse duration (approx)
    tau = np.sqrt(tau_0**2 + (2*spec_phase_coeffs[0]/tau_0)**2)
    
    t_start = t_0 + z_0/c
    t_end = 1e5*tau_0
    # number of time steps per laser period
    n = np.maximum(50, np.round(np.sqrt(P*tau_0/(tau*w_00**2))/(5e10)))  # empirically chosen resolution based on field strength
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

    # frequency dependent beam parameter based on Porras factor g_0, or arbitrary coefficients p_n
    if len(g_0_coeffs) == 1:
        z_R = z_R0*(omega_0/omega)**(g_0_coeffs)
    else:
        z_R = z_R0 + np.zeros(shape=len(omega))
        temp = 1
        for j in range(0, len(g_0_coeffs)):
            temp = temp*(j+1)
            z_R = z_R+z_R0*(g_0_coeffs[j]/temp)*((omega-omega_0)/omega_0)**(j+1)

    # initialize empty arrays
    z = np.zeros(shape=(len(time)))
    beta = np.zeros(shape=(len(time)))
    deriv2 = np.zeros(shape=(len(time)))
    KE = np.zeros(shape=(len(time)))

    # Set initial conditions
    beta[0] = beta_0
    z[0] = beta[0]*c*time[0]+z_0
    k_stop = -1

    # do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):

        pulse_spec = pulse_prep*np.exp(1j*(2*np.arctan((z[k]-z_omega)/z_R)-omega*z[k]/c))/(z_R*(1+((z[k]-z_omega)/z_R)**2))
        pulse_time = np.sum(pulse_spec*np.exp(1j*omega*time[k]))*omega_step/(delta_omega*np.sqrt(np.pi))
        field_total = Amp*np.exp(1j*Psi_0)*pulse_time

        deriv2[k] = (-q_e*np.real(field_total)*((1-beta[k]**2)**(3/2))/(m_e*c))  # Force in z

        if k==0:
            z[k+1] = z[k] + dt*c*beta[k]
            beta[k+1] = beta[k] + dt*deriv2[k]
        elif k==1:
            z[k+1] = z[k] + dt*c*(1.5*beta[k]-0.5*beta[k-1])
            beta[k+1] = beta[k] + dt*(1.5*deriv2[k]-0.5*deriv2[k-1])
        elif k==2:
            z[k+1] = z[k] + dt*c*((23/12)*beta[k]-(4/3)*beta[k-1]+(5/12)*beta[k-2])
            beta[k+1] = beta[k] + dt*((23/12)*deriv2[k]-(4/3)*deriv2[k-1]+(5/12)*deriv2[k-2])
        elif k==3:
            z[k+1] = z[k] + dt*c*((55/24)*beta[k]-(59/24)*beta[k-1]+(37/24)*beta[k-2]-(3/8)*beta[k-3])
            beta[k+1] = beta[k] + dt*((55/24)*deriv2[k]-(59/24)*deriv2[k-1]+(37/24)*deriv2[k-2]-(3/8)*deriv2[k-3])
        else:
            z[k+1] = z[k] + dt*c*((1901/720)*beta[k]-(1387/360)*beta[k-1]+(109/30)*beta[k-2]-(637/360)*beta[k-3]+(251/720)*beta[k-4])
            beta[k+1] = beta[k] + dt*((1901/720)*deriv2[k]-(1387/360)*deriv2[k-1]+(109/30)*deriv2[k-2]-(637/360)*deriv2[k-3]+(251/720)*deriv2[k-4])

        KE[k+1] = ((1/np.sqrt(1-beta[k+1]**2))-1)*m_e*c**2/q_e
        
        if (time[k] > 300*tau_0 and np.mean(np.abs(np.diff(KE[k-np.int(10*n):k+1]))/(KE[k+1]*dt)) < 1e7):
            k_stop = k+1
            break

    return time[:k_stop], z[:k_stop], v_z[:k_stop], KE[:k_stop]
