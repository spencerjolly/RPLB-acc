import numpy as np
from numba import jit

@jit(nopython=True)
def RPLB_acc_NoSTC_adaptive(lambda_0, tau_0, w_0, P, Psi_0, phi_2, t_0, z_0, beta_0):
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
    # stretched pulse duration
    tau = np.sqrt(tau_0**2 + (2*phi_2/tau_0)**2)
    
    t_start = t_0/(1-beta_0) + z_0/c
    t_end = +1e5*tau_0
    # number of time steps per laser period
    n = (lambda_0/(0.8e-6))*np.maximum(10, np.round(np.sqrt(P*tau_0/(tau*w_0**2))/(3e10)))  # empirically chosen resolution based on field strength
    num_t = np.int_(np.round(n*(t_end-t_start)/(lambda_0/c)))
    time = np.linspace(t_start, t_end, num_t)
    dt0 = time[1]-time[0]

    # initialize empty arrays
    z = np.zeros(shape=(len(time)))
    beta = np.zeros(shape=(len(time)))
    KE = np.zeros(shape=(len(time)))
    dt = np.zeros(shape=(len(time)))

    deriv2 = np.zeros(shape=(6))
    z_AB = np.zeros(shape=(6))
    beta_AB = np.zeros(shape=(6))
    time_AB = np.zeros(shape=(6))

    # Set initial conditions
    beta[0] = beta_0
    z[0] = beta_0*c*time[0] + z_0*(1-beta_0)
    dt[0] = dt0
    i_stop = -1

    #do 5th order Adams-Bashforth finite difference method
    for i in range(0, len(time)-1):
        z_AB[0] = z[i]
        beta_AB[0] = beta[i]
        time_AB[0] = time[i]
        dt_AB = dt[i]/5

        for k in range(0, 5):

            field_temp = np.exp(1j*(Psi_0+2*np.arctan(z_AB[k]/z_R)+omega_0*time_AB[k]-omega_0*z_AB[k]/c))/(z_R*(1+(z_AB[k]/z_R)**2))
            env_temp = np.exp(-((time_AB[k]-z_AB[k]/c)/tau)**2)
            temp_phase = np.exp(1j*(2*phi_2/(tau_0**4+(2*phi_2)**2))*(time_AB[k]-z_AB[k]/c)**2)
            field_total = Amp*(tau_0/tau)*field_temp*env_temp*temp_phase
            deriv2[k] = (-q_e*np.real(field_total)*((1-beta_AB[k]**2)**(3/2))/(m_e*c))

            time_AB[k+1] = time_AB[k] + dt_AB

            if k==0:
                z_AB[k+1] = z_AB[k] + dt_AB*c*beta_AB[k]
                beta_AB[k+1] = beta_AB[k] + dt_AB*deriv2[k]
            elif k==1:
                z_AB[k+1] = z_AB[k] + dt_AB*c*(1.5*beta_AB[k]-0.5*beta_AB[k-1])
                beta_AB[k+1] = beta_AB[k] + dt_AB*(1.5*deriv2[k]-0.5*deriv2[k-1])
            elif k==2:
                z_AB[k+1] = z_AB[k] + dt_AB*c*((23/12)*beta_AB[k]-(4/3)*beta_AB[k-1]+(5/12)*beta_AB[k-2])
                beta_AB[k+1] = beta_AB[k] + dt_AB*((23/12)*deriv2[k]-(4/3)*deriv2[k-1]+(5/12)*deriv2[k-2])
            elif k==3:
                z_AB[k+1] = z_AB[k] + dt_AB*c*((55/24)*beta_AB[k]-(59/24)*beta_AB[k-1]+(37/24)*beta_AB[k-2]-(3/8)*beta_AB[k-3])
                beta_AB[k+1] = beta_AB[k] + dt_AB*((55/24)*deriv2[k]-(59/24)*deriv2[k-1]+(37/24)*deriv2[k-2]-(3/8)*deriv2[k-3])
            else:
                z_AB[k+1] = z_AB[k] + dt_AB*c*((1901/720)*beta_AB[k]-(1387/360)*beta_AB[k-1]+(109/30)*beta_AB[k-2]-(637/360)*beta_AB[k-3]+(251/720)*beta_AB[k-4])
                beta_AB[k+1] = beta_AB[k] + dt_AB*((1901/720)*deriv2[k]-(1387/360)*deriv2[k-1]+(109/30)*deriv2[k-2]-(637/360)*deriv2[k-3]+(251/720)*deriv2[k-4])

        dt[i+1] = dt0
        time[i+1] = time[i] + dt[i]
        z[i+1] = z_AB[5]
        beta[i+1] = beta_AB[5]
        KE[i+1] = ((1/np.sqrt(1-beta[i+1]**2))-1)*m_e*c**2/q_e
        
        #if (time[i] > 300*tau_0 and np.mean(np.abs(np.diff(KE[i-np.int_(10*n):i+1]))/(KE[i+1]*dt)) < 1e7):
        #    i_stop = i+1
        #    break

    return time[:i_stop], z[:i_stop], beta[:i_stop], KE[:i_stop]
