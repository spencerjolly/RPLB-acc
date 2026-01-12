import numpy as np
from numba import jit

@jit(nopython=True)
def RPLB_acc_PulseFront_arbitrary(lambda_0, tau_0, a, P, SP, PM, PF, phi_2, t_0, z_0, beta_0):
    # initialize constants (SI units)
    c = 2.99792458e8  # speed of light
    m_e = 9.10938356e-31
    q_e = 1.60217662e-19
    e_0 = 8.85418782e-12
    # calculate frequency properties
    omega_0 = 2*np.pi*c/lambda_0
    k_0 = omega_0/c
    delta_omega = 2/tau_0
    # amplitude factor
    Amp = np.sqrt(8*P/(np.pi*e_0*c))
    # stretched pulse duration
    tau = np.sqrt(tau_0**2 + (2*phi_2/tau_0)**2)
    
    t_start = t_0/(1-beta_0) + z_0/c
    t_end = +1e5*tau_0
    # number of time steps per laser period
    n = (lambda_0/(0.8e-6))*np.maximum(50, np.round(np.sqrt(P*tau_0*k_0/(tau*2*a))/(3e10)))
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
    z[0] = beta_0*c*time[0] + z_0*(1-beta_0)
    k_stop = -1

    # do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):
        
        alpha = np.linspace(0, np.sqrt(9*2/(k_0*a)), np.int_(201+2*np.round(np.abs(z[k])/a)))
        d_alpha = alpha[1]-alpha[0]
        scaling = np.sqrt(2*k_0*a)*np.tan(alpha/2)
        illum = scaling*np.exp(-scaling**2)
        spatial_profile = 1.0*SP[0] + \
                          (2 - 2*scaling**2)*SP[1]/np.sqrt(1+1) + \
                          (((2*scaling**2)**2 - 6*(2*scaling**2) + 6)/2)*SP[2]/np.sqrt(2+1) + \
                          ((-1*(2*scaling**2)**3 + 12*(2*scaling**2)**2 - 36*(2*scaling**2) + 24)/6)*SP[3]/np.sqrt(3+1) + \
                          (((2*scaling**2)**4 - 20*(2*scaling**2)**3 + 120*(2*scaling**2)**2 - 240*(2*scaling**2) + 120)/24)*SP[4]/np.sqrt(4+1) + \
                          ((-1*(2*scaling**2)**5 + 30*(2*scaling**2)**4 - 300*(2*scaling**2)**3 + 1200*(2*scaling**2)**2 - 1800*(2*scaling**2) + 720)/120)*SP[5]/np.sqrt(5+1) + \
                          (((2*scaling**2)**6 - 42*(2*scaling**2)**5 + 630*(2*scaling**2)**4 - 4200*(2*scaling**2)**3 + 12600*(2*scaling**2)**2 - 15120*(2*scaling**2) + 5040)/720)*SP[6]/np.sqrt(6+1)
        phase = omega_0*time[k] - k_0*z[k]*np.cos(alpha) + \
                PM[0] + PM[1]*scaling + \
        		PM[2]*scaling**2 + PM[3]*scaling**3 + \
        		PM[4]*scaling**4 + PM[5]*scaling**5 + \
        		PM[6]*scaling**6 + PM[7]*scaling**7 + \
        		PM[8]*scaling**8
        delay = PF[0]*scaling + \
                PF[1]*scaling**2 + PF[2]*scaling**3 + \
                PF[3]*scaling**4 + PF[4]*scaling**5 + \
                PF[5]*scaling**6 + PF[6]*scaling**7 + \
                PF[7]*scaling**8
        apod = (1/np.cos(alpha/2))**(2)

        integrand = np.sin(alpha)**2

        corr = np.sqrt(k_0)*k_0*np.sqrt(a)/np.sqrt(2)

        field_temp = np.sum(d_alpha*np.exp(-(((phase-PM[0])/omega_0 - delay)/tau)**2)*corr*illum*spatial_profile*np.exp(1j*phase)*apod*integrand)

        temp_phase = np.exp(1j*(2*phi_2/(tau_0**4+(2*phi_2)**2))*(time[k]-z[k]/c)**2)
        field_total = Amp*(tau_0/tau)*field_temp*temp_phase
        
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
        
        if (time[k] > 300*tau_0 and np.mean(np.abs(np.diff(KE[k-np.int_(10*n):k+1]))/(KE[k+1]*dt)) < 1e7):
            k_stop = k+1
            break

    return time[:k_stop], z[:k_stop], beta[:k_stop], KE[:k_stop]
