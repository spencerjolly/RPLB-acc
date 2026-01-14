import numpy as np
from numba import jit

@jit(nopython=True)
def RPLB_acc_LC_ponderomotive(lambda_0, tau_0, w_0, P, phi_2, t_0, z_0, beta_0, tau_p):
    # initialize constants (SI units)
    c = 2.99792458e8  # speed of light
    m_e = 9.10938356e-31
    q_e = 1.60217662e-19
    e_0 = 8.85418782e-12
    # calculate frequency properties
    omega_0 = 2*np.pi*c/lambda_0
    delta_omega = 2/tau_0
    # calculate Rayleigh range
    z_R = (omega_0*w_0**2)/(2*c)
    # amplitude factor
    Amp = np.sqrt(8*P/(np.pi*e_0*c))
    Amp2 = np.sqrt(2/np.abs(phi_2))/delta_omega
    # stretched pulse duration
    tau = np.sqrt(tau_0**2 + (2*phi_2/tau_0)**2)
    
    t_start = t_0/(1-beta_0) + z_0/c
    t_end = +1e5*tau_0
    # number of time steps per laser period
    n = (lambda_0/(0.8e-6))*np.maximum(5, np.round(np.sqrt(P*tau_0/(tau*w_0**2))/(5e11)))  # empirically chosen resolution based on field strength
    num_t = np.int_(np.round(n*(t_end-t_start)/(lambda_0/c)))
    time = np.linspace(t_start, t_end, num_t)
    dt = time[1]-time[0]

    # initialize empty arrays
    z = np.zeros(shape=(len(time)))
    beta = np.zeros(shape=(len(time)))
    deriv2 = np.zeros(shape=(len(time)))
    KE = np.zeros(shape=(len(time)))

    # Set initial conditions
    beta[0] = beta_0
    KE[0] = ((1/np.sqrt(1-beta[0]**2))-1)*m_e*c**2/q_e
    z[0] = beta_0*c*time[0] + z_0*(1-beta_0)
    k_stop = -1

    #do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):
        I_z = (1/z_R**2)*np.exp(-2*(time[k]/(phi_2*delta_omega) - z[k]/(c*phi_2*delta_omega))**2)/(1 + (z[k]*(1/z_R + tau_p/(c*phi_2)) - time[k]*tau_p/phi_2)**2)**2
        I_z_deriv1 = -4*((z[k]*(1/z_R + tau_p/(c*phi_2)) - time[k]*tau_p/phi_2)*(1/z_R + tau_p/(c*phi_2))/(1 + (z[k]*(1/z_R + tau_p/(c*phi_2)) - time[k]*tau_p/phi_2)**2))
        I_z_deriv2 = 4*(time[k]/(phi_2*delta_omega) - z[k]/(c*phi_2*delta_omega))/(c*phi_2*delta_omega)

        omega_bar = omega_0 + (time[k] - z[k]/c)/(phi_2) - (tau_p/phi_2)/(1 + (z[k]*(1/z_R + tau_p/(c*phi_2)) - time[k]*tau_p/phi_2)**2)
        force = (-q_e**2/(4*m_e*(omega_bar)**2))*(Amp**2 * Amp2**2)*I_z*(I_z_deriv1 + I_z_deriv2)
        deriv2[k] = (force*((1 - beta[k]**2)**(3/2))/(m_e*c))

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
        
        if (time[k] > 300*tau_0 and np.mean(np.abs(np.diff(KE[k-np.int_(10*n):k+1]))/(KE[k+1]*dt)) < 1e6):
            k_stop = k+1
            break

    return time[:k_stop], z[:k_stop], beta[:k_stop], KE[:k_stop]