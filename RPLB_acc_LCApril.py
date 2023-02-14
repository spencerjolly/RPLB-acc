import numpy as np
from numba import jit

@jit(nopython=True)
def RPLB_acc_LCApril(lambda_0, s, a, P, Psi_0, phi_2, phi_3, t_0, z_0, beta_0, tau_p):
    # initialize constants (SI units)
    c = 2.99792458e8  # speed of light
    m_e = 9.10938356e-31
    q_e = 1.60217662e-19
    e_0 = 8.85418782e-12
    # calculate frequency properties
    omega_0 = 2*np.pi*c/lambda_0
    tau_0 = s*np.sqrt(np.exp(2/(s+1))-1)/omega_0
    delta_omega = 2/tau_0
    # amplitude factor
    Amp = -1*np.sqrt(8*P/(np.pi*e_0*c))*a*c/(2*omega_0)
    
    t_start = t_0 + z_0/c
    t_end = +1e5*tau_0
    # number of time steps per laser period
    n = (lambda_0/(0.8e-6))*50  # np.maximum(50, np.round(np.sqrt(P/(w_0**2))/(5e10)))  # empirically chosen resolution based on field strength
    num_t = np.int_(np.round(n*(t_end-t_start)/(lambda_0/c)))
    time = np.linspace(t_start, t_end, num_t)
    dt = time[1]-time[0]
    
    omega = np.linspace((omega_0-4*delta_omega), (omega_0+4*delta_omega), 300)
    omega_step = omega[1]-omega[0]
    
    pulse_temp = np.exp(-((omega-omega_0)/delta_omega)**2)
    pulse_prep = pulse_temp*np.exp(-1j*((phi_2/2)*(omega-omega_0)**2 + (phi_3/6)*(omega-omega_0)**3))
    k = omega/c
    z_omega = (np.sqrt((k*a)**2 + 1) + k*a)*tau_p*(omega-omega_0)/(2*k)

    # initialize empty arrays
    z = np.zeros(shape=(len(time)))
    beta = np.zeros(shape=(len(time)))
    deriv2 = np.zeros(shape=(len(time)))
    KE = np.zeros(shape=(len(time)))

    # Set initial conditions
    beta[0] = beta_0
    z[0] = beta[0]*c*time[0]+z_0
    i_stop = -1

    #do 5th order Adams-Bashforth finite difference method
    for i in range(0, len(time)-1):

        Rt = np.sqrt((z[i] - z_omega + 1j*a)**2)
        pulse_spec = pulse_prep*(2*2*1j*Amp*np.exp(-k*a)*np.exp(-1j*k*z_omega)/(Rt)**2)*(np.sin(k*Rt)/Rt-k*np.cos(k*Rt))
        pulse_time = np.sum(pulse_spec*np.exp(1j*omega*time[i]))*omega_step/(delta_omega*np.sqrt(np.pi))
        field_total = np.exp(1j*(Psi_0+np.pi/2))*pulse_time
        
        deriv2[i] = (-q_e*np.real(field_total)*((1-beta[i]**2)**(3/2))/(m_e*c))

        if i==0:
            z[i+1] = z[i] + dt*c*beta[i]
            beta[i+1] = beta[i] + dt*deriv2[i]
        elif i==1:
            z[i+1] = z[i] + dt*c*(1.5*beta[i]-0.5*beta[i-1])
            beta[i+1] = beta[i] + dt*(1.5*deriv2[i]-0.5*deriv2[i-1])
        elif i==2:
            z[i+1] = z[i] + dt*c*((23/12)*beta[i]-(4/3)*beta[i-1]+(5/12)*beta[i-2])
            beta[i+1] = beta[i] + dt*((23/12)*deriv2[i]-(4/3)*deriv2[i-1]+(5/12)*deriv2[i-2])
        elif i==3:
            z[i+1] = z[i] + dt*c*((55/24)*beta[i]-(59/24)*beta[i-1]+(37/24)*beta[i-2]-(3/8)*beta[i-3])
            beta[i+1] = beta[i] + dt*((55/24)*deriv2[i]-(59/24)*deriv2[i-1]+(37/24)*deriv2[i-2]-(3/8)*deriv2[i-3])
        else:
            z[i+1] = z[i] + dt*c*((1901/720)*beta[i]-(1387/360)*beta[i-1]+(109/30)*beta[i-2]-(637/360)*beta[i-3]+(251/720)*beta[i-4])
            beta[i+1] = beta[i] + dt*((1901/720)*deriv2[i]-(1387/360)*deriv2[i-1]+(109/30)*deriv2[i-2]-(637/360)*deriv2[i-3]+(251/720)*deriv2[i-4])

        KE[i+1] = ((1/np.sqrt(1-beta[i+1]**2))-1)*m_e*c**2/q_e
        
        if (time[i] > 300*tau_0 and np.mean(np.abs(np.diff(KE[i-np.int(10*n):i+1]))/(KE[i+1]*dt)) < 1e7):
            i_stop = i+1
            break

    return time[:i_stop], z[:i_stop], beta[:i_stop], KE[:i_stop]
