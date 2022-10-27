import numpy as np
from numba import jit

@jit(nopython=True)
def RPLB_acc_NoSTC_2D(lambda_0, tau_0, w_0, P, Psi_0, phi_2, t_0, z_0, r_0, beta_0):
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
    # perturbation parameter
    eps = w_0/z_R
    # amplitude factor
    P_corr = 1 + 3*(eps/2)**2 + 9*(eps/2)**4
    Amp = np.sqrt(8*P/(P_corr*np.pi*e_0*c)) * (omega_0/(2*c))
    # stretched pulse duration
    tau = np.sqrt(tau_0**2 + (2*phi_2/tau_0)**2)
    
    t_start = t_0 + z_0/c
    t_end = +1e5*tau_0
    # number of time steps per laser period
    n = (lambda_0/(0.8e-6))*200  # np.maximum(50, np.round(np.sqrt(P/(w_0**2))/(5e10)))  # empirically chosen resolution based on field strength
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

    z[0] = beta_0*c*time[0] + z_0
    r[0] = r_0
    v_z[0] = beta_0*c
    v_r[0] = 0.0
    gamma[0] = 1/np.sqrt(1-beta_0**2)
    KE[0] = (gamma[0]-1)*m_e*c**2/q_e
    k_stop = -1

    # do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):
        
        rho = r[k]/w_0
        
        phi_G = np.arctan(z[k]/z_R)
        w = w_0*np.sqrt(1+(z[k]/z_R)**2)
        R_inv = z[k]/(z[k]**2 + z_R**2)
        phi_norm = Psi_0-(omega_0/c)*(z[k]+(R_inv*(rho*w_0)**2)/2)+omega_0*time[k]
        trans = np.exp(-(rho*w_0/w)**2)

        c_2 = (w_0/w)**2 * np.exp(1j*(phi_norm + 2*phi_G))
        c_3 = (w_0/w)**3 * np.exp(1j*(phi_norm + 3*phi_G))
        c_4 = (w_0/w)**4 * np.exp(1j*(phi_norm + 4*phi_G))
        c_5 = (w_0/w)**5 * np.exp(1j*(phi_norm + 5*phi_G))
        c_6 = (w_0/w)**6 * np.exp(1j*(phi_norm + 6*phi_G))
        c_7 = (w_0/w)**7 * np.exp(1j*(phi_norm + 7*phi_G))
        c_8 = (w_0/w)**8 * np.exp(1j*(phi_norm + 8*phi_G))

        env_temp = np.exp(-((phi_norm-Psi_0)/(omega_0*tau))**2)
        temp_phase = np.exp(1j*(2*phi_2/(tau_0**4+(2*phi_2)**2))*(time[k]-z[k]/c)**2)
        pulse_prep = (tau_0/tau)*env_temp*temp_phase

        E_z_time = pulse_prep*((c_2 - c_3*rho**2)*eps**2 +
                               ((1/2)*c_3 + (1/2)*c_4*rho**2 - (5/4)*c_5*rho**4 + (1/4)*c_6*rho**6)*eps**4)

        E_r_time = pulse_prep*((c_2)*eps +
                               (-(1/2)*c_3 + c_4*rho**2 - (1/4)*c_5*rho**4)*eps**3 +
                               (-(3/8)*c_4 - (3/8)*c_5*rho**2 + (17/16)*c_6*rho**4 -
                                (3/8)*c_7*rho**6 + (1/32)*c_8*rho**8)*eps**5)*np.exp(+1j*np.pi/2)*rho

        B_t_time = pulse_prep*((c_2)*eps +
                               ((1/2)*c_3 + (1/2)*c_4*rho**2 - (1/4)*c_5*rho**4)*eps**3 +
                               ((3/8)*c_4 + (3/8)*c_5*rho**2 + (3/16)*c_6*rho**4 -
                                (1/4)*c_7*rho**6 + (1/32)*c_8*rho**8)*eps**5)*np.exp(+1j*np.pi/2)*rho/c

        E_z_total = np.real(Amp*trans*E_z_time)
        E_r_total = np.real(Amp*trans*E_r_time)
        dot_product = v_z[k]*E_z_total + v_r[k]*E_r_total
        B_t_total = np.real(Amp*trans*B_t_time)

        deriv2[k] = (-q_e/(gamma[k]*m_e))*(E_z_total+v_r[k]*B_t_total-v_z[k]*dot_product/(c**2))  # Force in z
        deriv4[k] = (-q_e/(gamma[k]*m_e))*(E_r_total-v_z[k]*B_t_total-v_r[k]*dot_product/(c**2))  # Force in r

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
           
        if (time[k] > 300*tau_0 and np.mean(np.abs(np.diff(KE[k-np.int(10*n):k+1]))/(KE[k+1]*dt)) < 1e7):
            k_stop = k+1
            break

    return time[:k_stop], z[:k_stop], r[:k_stop], v_z[:k_stop], v_r[:k_stop], KE[:k_stop]