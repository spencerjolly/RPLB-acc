import numpy as np
from numba import jit

@jit(nopython=True)
def RPLB_acc_anySTC_arbitrary(lambda_0, tau_0, a, P, PM, t_0, z_0, beta_0):
    # initialize constants (SI units)
    c = 2.99792458e8  # speed of light
    m_e = 9.10938356e-31
    q_e = 1.60217662e-19
    e_0 = 8.85418782e-12
    # calculate frequency properties
    omega_0 = 2*np.pi*c/lambda_0
    delta_omega = 2/tau_0
    # amplitude factor
    Amp = np.sqrt(8*P/(np.pi*e_0*c))
    # stretched pulse duration (approx)
    tau = np.sqrt(tau_0**2 + (2*PM[2,0]/tau_0)**2)
    
    t_start = t_0 + z_0/(c*(1-beta_0))
    t_end = +1e5*tau_0
    # number of time steps per laser period
    n = 50
    num_t = np.int_(np.round(n*(t_end-t_start)/(lambda_0/c)))
    time = np.linspace(t_start, t_end, num_t)
    dt = time[1]-time[0]

    omega = np.linspace((omega_0-4*delta_omega), (omega_0+4*delta_omega), 300)
    omega_step = omega[1]-omega[0]
    pulse_mid = np.zeros(shape=(len(omega)), dtype=np.complex128)

    pulse_temp = np.exp(-((omega-omega_0)/delta_omega)**2)  # spectral envelope

    # initialize empty arrays
    z = np.zeros(shape=(len(time)))
    beta = np.zeros(shape=(len(time)))
    deriv2 = np.zeros(shape=(len(time)))
    KE = np.zeros(shape=(len(time)))

    # Set initial conditions
    beta[0] = beta_0
    z[0] = beta_0*c*time[0] + z_0
    k_stop = -1

    # do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):
        
        for ii in range(0, len(omega)):
            
            alpha = np.linspace(0, 1.0, 501)
            d_alpha = alpha[1]-alpha[0]
            k_real = omega[ii]/c
            scaling = np.sqrt(2*k_real*a)*np.tan(alpha/2)
            illum = scaling*np.exp(-scaling**2)
            phase = ((PM[0,0] + PM[0,1]*scaling + PM[0,2]*scaling**2 + PM[0,3]*scaling**3 + PM[0,4]*scaling**4) + \
                     (PM[1,0] + PM[1,1]*scaling + PM[1,2]*scaling**2 + PM[1,3]*scaling**3 + PM[1,4]*scaling**4)*(omega[ii]-omega_0) + \
                     (PM[2,0] + PM[2,1]*scaling + PM[2,2]*scaling**2 + PM[2,3]*scaling**3 + PM[2,4]*scaling**4)*((omega[ii]-omega_0)**2)/2 + \
                     (PM[3,0] + PM[3,1]*scaling + PM[3,2]*scaling**2 + PM[3,3]*scaling**3 + PM[3,4]*scaling**4)*((omega[ii]-omega_0)**3)/6 + \
                     (PM[4,0] + PM[4,1]*scaling + PM[4,2]*scaling**2 + PM[4,3]*scaling**3 + PM[4,4]*scaling**4)*((omega[ii]-omega_0)**4)/24)
                        
            apod = (1/np.cos(alpha/2))**(2)

            integrand1 = np.exp(-1j*k_real*z[k]*np.cos(alpha))
            integrand2 = np.sin(alpha)**2

            corr = np.sqrt(k_real)*k_real*np.sqrt(a)/np.sqrt(2)

            pulse_mid[ii] = np.sum(d_alpha*corr*illum*np.exp(1j*phase)*apod*integrand1*integrand2)

        pulse_spec = pulse_temp*pulse_mid
        pulse_time = np.sum(pulse_spec*np.exp(1j*omega*time[k]))*omega_step/(delta_omega*np.sqrt(np.pi))
        field_total = Amp*pulse_time

        deriv2[k] = (-q_e*np.real(field_total)*((1-beta[k]**2)**(3/2))/(m_e*c))  # Lorentz force in z

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

    return time[:k_stop], z[:k_stop], beta[:k_stop], KE[:k_stop]
