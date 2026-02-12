import numpy as np
from numba import jit

@jit(nopython=True)
def RPLB_acc_diffraction(lambda_0, tau_0, a, P, PM_type, PM, PF, phi_2, t_0, z_0, beta_0):
    """
    Function to accelerate an on-axis electron when under the influence of the electric field of
    a tightly focused radially-polarized laser beam (RPLB) of ultrashort duration. The electric field
    is calculated using a vector diffraction integral.

    Parameters
    ----------
    lambda_0 = central wavelength of the laser pulse [m]
    tau_0 = Fourier-limited 1/exp(2) pulse duration of the laser pulse [s]
    a = confocal parameter, related to the tightness of focusing
    P = Fourier-limited pulse power in the absence of aberrations or pulse-front delay
    PM_type = type of phase map. 0 = generic polynomial, 1 = Zernike
    PM = phase map
    PF = pulse-front delay
    phi_2 = group-delay dispersion
    t_0 = initial starting time of the simulation in terms of the pulse peak relative to the electron starting position [s]
    z_0 = electron starting position [m]
    beta_0 = initial electron speed beta=v/c
    """
    # initialize constants (SI units)
    c = 2.99792458e8  # speed of light [m/s]
    m_e = 9.10938356e-31  # electron mass [kg]
    q_e = 1.60217662e-19  # electron charge [C]
    e_0 = 8.85418782e-12  # permittivity of free space
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

    corr = np.sqrt(k_0)*k_0*np.sqrt(a)/np.sqrt(2)

    # do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):
        
        alpha = np.linspace(0, np.sqrt(9*2/(k_0*a)), np.int_(201+2*np.round(np.abs(z[k])/a)))
        d_alpha = alpha[1]-alpha[0]
        scaling = np.sqrt(2*k_0*a)*np.tan(alpha/2)
        illum = scaling*np.exp(-scaling**2)

        if PM_type==0:
            phase = omega_0*time[k] - k_0*z[k]*np.cos(alpha) + \
                    PM[0] + PM[1]*scaling + \
            		PM[2]*scaling**2 + PM[3]*scaling**3 + \
            		PM[4]*scaling**4 + PM[5]*scaling**5 + \
            		PM[6]*scaling**6 + PM[7]*scaling**7 + \
            		PM[8]*scaling**8
        elif PM_type = 1:
            phase = omega_0*time[k] - k_0*z[k]*np.cos(alpha) + PM[0] + \
                PM[1]*np.sqrt(3)*(2*scaling**2 - 1) + \
                PM[2]*np.sqrt(5)*(6*scaling**4 - 6*scaling**2 + 1) + \
                PM[3]*np.sqrt(7)*(20*scaling**6 - 30*scaling**4 + 12*scaling**2 - 1)

        delay = PF[0]*scaling + \
                PF[1]*scaling**2 + PF[2]*scaling**3 + \
                PF[3]*scaling**4 + PF[4]*scaling**5 + \
                PF[5]*scaling**6 + PF[6]*scaling**7 + \
                PF[7]*scaling**8
        apod = (1/np.cos(alpha/2))**(2)

        integrand = np.sin(alpha)**2

        field_temp = np.sum(d_alpha*np.exp(-((phase-PM[0])/omega_0 - delay)**2/(tau_0**2 + 2*1j*phi_2))*corr*illum*np.exp(1j*phase)*apod*integrand)

        field_total = Amp*(tau_0/tau)*field_temp
        
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