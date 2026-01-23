import numpy as np
from numba import jit

@jit(nopython=True)
def RPLB_acc_LC_2D_ponderomotive(lambda_0, tau_0, w_0, P, phi_2, t_0, z_0, r_0, beta_0, tau_p):
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
    eps = w_0/z_R
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
    r = np.zeros(shape=(len(time)))
    v_z = np.zeros(shape=(len(time)))
    v_r = np.zeros(shape=(len(time)))
    gamma = np.zeros(shape=(len(time)))
    KE = np.zeros(shape=(len(time)))
    deriv2 = np.zeros(shape=(len(time)))
    deriv4 = np.zeros(shape=(len(time)))

    # Set initial conditions
    z[0] = beta_0*c*time[0] + z_0*(1-beta_0)
    r[0] = r_0
    v_z[0] = beta_0*c
    v_r[0] = 0.0
    gamma[0] = 1/np.sqrt(1-beta_0**2)
    KE[0] = (gamma[0]-1)*m_e*c**2/q_e
    k_stop = -1

    #do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):
        t_prime = time[k] - z[k]/c
        rho = r[k]/w_0
        Qff2 = 1/(1 + (z[k]*(1/z_R + tau_p/(c*phi_2)) - time[k]*tau_p/phi_2)**2)
        env = np.exp(-2*(t_prime/(phi_2*delta_omega))**2)*np.exp(-2*rho**2 * Qff2)

        I_z = (1/z_R**2)*env*Qff2**2 * (1 - 2*rho**2 * Qff2 + rho**4 * Qff2**2)
        I_r = (1/w_0**2)*env*Qff2**2 * rho**2

        denv_dz = 4*(t_prime/(phi_2*delta_omega))*(1/(c*phi_2*delta_omega))
        dQff2_dz = -2*(z[k]*(1/z_R + tau_p/(c*phi_2)) - time[k]*tau_p/phi_2)*(1/z_R + tau_p/(c*phi_2))*Qff2**2

        dI_z_dz = I_z*(denv_dz - 2*rho**2 * dQff2_dz + 2*dQff2_dz/Qff2) + (1/z_R**2)*env*Qff2**2 *(-2*rho**2 * dQff2_dz + 2*rho**4 * Qff2*dQff2_dz)
        dI_r_dz = I_r*(denv_dz - 2*rho**2 * dQff2_dz + 2*dQff2_dz/Qff2)

        dI_z_dr = (1/w_0)*(I_z*(-4*rho*Qff2) + (1/z_R**2)*env*Qff2**2 *(-4*rho*Qff2 + 4*rho**3 * Qff2**2))
        dI_r_dr = (1/w_0)*(I_r*(-4*rho*Qff2) + 2*(1/w_0**2)*env*Qff2**2 * rho)

        omega_bar = omega_0 + (time[k] - z[k]/c)/(phi_2) - (tau_p/phi_2)/(1 + (z[k]*(1/z_R + tau_p/(c*phi_2)) - time[k]*tau_p/phi_2)**2)

        force_z = (-q_e**2/(4*m_e*(omega_bar)**2))*(Amp**2 * Amp2**2)*(dI_z_dz + dI_r_dz)
        force_r = (-q_e**2/(4*m_e*(omega_bar)**2))*(Amp**2 * Amp2**2)*(dI_z_dr + dI_r_dr)

        deriv2[k] = force_z/(gamma[k]*m_e)
        deriv4[k] = force_r/(gamma[k]*m_e)

        if k==0:
            z[k+1] = z[k] + dt*v_z[k]
            v_z[k+1] = v_z[k] + dt*deriv2[k]
            r[k+1] = r[k] + dt*v_r[k]
            v_r[k+1] = v_r[k] + dt*deriv4[k]
        elif k==1:
            z[k+1] = z[k] + dt*(1.5*v_z[k]-0.5*v_z[k-1])
            v_z[k+1] = v_z[k] + dt*(1.5*deriv2[k]-0.5*deriv2[k-1])
            r[k+1] = r[k] + dt*(1.5*v_r[k]-0.5*v_r[k-1])
            v_r[k+1] = v_r[k] + dt*(1.5*deriv4[k]-0.5*deriv4[k-1])
        elif k==2:
            z[k+1] = z[k] + dt*((23/12)*v_z[k]-(4/3)*v_z[k-1]+(5/12)*v_z[k-2])
            v_z[k+1] = v_z[k] + dt*((23/12)*deriv2[k]-(4/3)*deriv2[k-1]+(5/12)*deriv2[k-2])
            r[k+1] = r[k] + dt*((23/12)*v_r[k]-(4/3)*v_r[k-1]+(5/12)*v_r[k-2])
            v_r[k+1] = v_r[k] + dt*((23/12)*deriv4[k]-(4/3)*deriv4[k-1]+(5/12)*deriv4[k-2])
        elif k==3:
            z[k+1] = z[k] + dt*((55/24)*v_z[k]-(59/24)*v_z[k-1]+(37/24)*v_z[k-2]-(3/8)*v_z[k-3])
            v_z[k+1] = v_z[k] + dt*((55/24)*deriv2[k]-(59/24)*deriv2[k-1]+(37/24)*deriv2[k-2]-(3/8)*deriv2[k-3])
            r[k+1] = r[k] + dt*((55/24)*v_r[k]-(59/24)*v_r[k-1]+(37/24)*v_r[k-2]-(3/8)*v_r[k-3])
            v_r[k+1] = v_r[k] + dt*((55/24)*deriv4[k]-(59/24)*deriv4[k-1]+(37/24)*deriv4[k-2]-(3/8)*deriv4[k-3])
        else:
            z[k+1] = z[k] + dt*((1901/720)*v_z[k]-(1387/360)*v_z[k-1]+(109/30)*v_z[k-2]-(637/360)*v_z[k-3]+(251/720)*v_z[k-4])
            v_z[k+1] = v_z[k] + dt*((1901/720)*deriv2[k]-(1387/360)*deriv2[k-1]+(109/30)*deriv2[k-2]-(637/360)*deriv2[k-3]+(251/720)*deriv2[k-4])
            r[k+1] = r[k] + dt*((1901/720)*v_r[k]-(1387/360)*v_r[k-1]+(109/30)*v_r[k-2]-(637/360)*v_r[k-3]+(251/720)*v_r[k-4])
            v_r[k+1] = v_r[k] + dt*((1901/720)*deriv4[k]-(1387/360)*deriv4[k-1]+(109/30)*deriv4[k-2]-(637/360)*deriv4[k-3]+(251/720)*deriv4[k-4])

        gamma[k+1] = 1/np.sqrt(1-(v_z[k+1]**2+v_r[k+1]**2)/c**2)
        KE[k+1] = (gamma[k+1]-1)*m_e*c**2/q_e
           
        if (time[k] > 300*tau_0 and np.mean(np.abs(np.diff(KE[k-np.int_(10*n):k+1]))/(KE[k+1]*dt)) < 1e7):
            k_stop = k+1
            break

    return time[:k_stop], z[:k_stop], r[:k_stop], v_z[:k_stop], v_r[:k_stop], KE[:k_stop]