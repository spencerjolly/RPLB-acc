import numpy as np
from scipy.special import erfc
#from numba.extending import get_cython_function_address
from numba import jit, njit
#import ctypes

#addr = get_cython_function_address("scipy.special.cython_special", "erfc")
#functype = ctypes.CFUNCTYPE(ctypes.c_double_complex, ctypes.c_double_complex)
#erfc_fn = functype(addr)

@njit
def call_erfc(x):
    return erfc(x)

@jit(nopython=True)
def RPLB_acc_NoSTC_arbitrary_analytical(lambda_0, tau_0, a, P, Psi_0, phi_2, t_0, z_0, beta_0, spher):
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
    
    t_start = t_0 + z_0/(c*(1-beta_0))
    t_end = +1e5*tau_0
    # number of time steps per laser period
    n = 50
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
    z[0] = beta_0*c*time[0] + z_0
    k_stop = -1

    # do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):
        
        term = (1-1j*z[k]/a)
        const = (2*np.exp(-1j*k_0*z[k])/a)/(16*spher**(3/2))
        E_z_spher = const*((1-1j)*np.sqrt(2*np.pi)*term*np.exp(1j*term**2/(4*spher))*call_erfc((term/2)*np.exp(1j*np.pi/4)*np.sqrt(1/(spher))) + 4*1j*np.sqrt(spher))

        field_temp = np.sum(np.exp(1j*Psi_0)*E_z_spher)

        env_temp = np.exp(-((time[k]-z[k]/c)/tau)**2)
        temp_phase = np.exp(1j*(2*phi_2/(tau_0**4+(2*phi_2)**2))*(time[k]-z[k]/c)**2)
        field_total = Amp*(tau_0/tau)*field_temp*env_temp*temp_phase
        
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
