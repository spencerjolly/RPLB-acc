import numpy as np
from numba import jit

@jit(nopython=True)
def RPLB_acc_LC_2D(lambda_0, tau_0, w_0, P, Psi_0, phi_2, phi_3, t_0, z_0, r_0, beta_0, tau_p):
    # initialize constants (SI units)
    c = 2.99792458e8 #speed of light
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
    P_corr = 1 + 3*(eps/2)**2 + 9*(eps/2)**4
    Amp = np.sqrt(8*P/(P_corr*np.pi*e_0*c)) * (omega_0/(2*c))
    # extended Rayleigh range
    z_R_e = tau_p*z_R*delta_omega
    
    t_start = t_0 + z_0/(c*(1-beta_0))
    t_end = +1e5*tau_0
    n = (lambda_0/(0.8e-6))*200  # number of time steps per laser period
    num_t = np.int_(np.round(n*(t_end-t_start)/(lambda_0/c)))
    time = np.linspace(t_start, t_end, num_t)
    dt = time[1]-time[0]

    omega = np.linspace((omega_0-4*delta_omega), (omega_0+4*delta_omega), 300)
    omega_step = omega[1]-omega[0]

    pulse_temp = np.exp(-((omega-omega_0)/delta_omega)**2)
    pulse_prep = pulse_temp*np.exp(-1j*((phi_2/2)*(omega-omega_0)**2 + (phi_3/6)*(omega-omega_0)**3))
    z_omega = z_R*tau_p*(omega-omega_0)

    z = np.empty(shape=(len(time)))
    r = np.empty(shape=(len(time)))
    v_z = np.empty(shape=(len(time)))
    v_r = np.empty(shape=(len(time)))
    gamma = np.empty(shape=(len(time)))
    KE = np.zeros(shape=(len(time)))
    deriv2 = np.empty(shape=(len(time)))
    deriv4 = np.empty(shape=(len(time)))

    z[0] = beta_0*c*time[0] + z_0*(1-beta_0)
    r[0] = r_0
    v_z[0] = beta_0*c
    v_r[0] = 0.0
    gamma[0] = 1/np.sqrt(1-beta_0**2)
    KE[0] = ((1/np.sqrt(1-beta_0**2))-1)*m_e*c**2/q_e
    k_stop = -1

    # do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):

        phi_G = np.arctan((z[k]-z_omega)/z_R)
        w = w_0*np.sqrt(1+((z[k]-z_omega)/z_R)**2)
        R_inv = (z[k]-z_omega)/((z[k]-z_omega)**2 + z_R**2)
        phi_norm = Psi_0-(omega/c)*(z[k]+(R_inv*r[k]**2)/2)+omega*time[k]
        trans = np.exp(-(r[k]/w)**2)

        c_2 = (w_0/w)**2 * np.exp(1j*(phi_norm + 2*phi_G))
        c_3 = (w_0/w)**3 * np.exp(1j*(phi_norm + 3*phi_G))
        c_4 = (w_0/w)**4 * np.exp(1j*(phi_norm + 4*phi_G))
        c_5 = (w_0/w)**5 * np.exp(1j*(phi_norm + 5*phi_G))
        c_6 = (w_0/w)**6 * np.exp(1j*(phi_norm + 6*phi_G))
        c_7 = (w_0/w)**7 * np.exp(1j*(phi_norm + 7*phi_G))
        c_8 = (w_0/w)**8 * np.exp(1j*(phi_norm + 8*phi_G))

        rho = r[k]/w_0

        E_z_spec = pulse_prep*((c_2 - c_3*rho**2)*eps**2 +
                               ((1/2)*c_3 + (1/2)*c_4*rho**2 - (5/4)*c_5*rho**4 + (1/4)*c_6*rho**6)*eps**4)
        E_z_time = np.sum(E_z_spec*trans)*omega_step/(delta_omega*np.sqrt(np.pi))

        E_r_spec = pulse_prep*((c_2*rho)*eps +
                               (-(1/2)*c_3*rho + c_4*rho**3 - (1/4)*c_5*rho**5)*eps**3 +
                               (-(3/8)*c_4*rho - (3/8)*c_5*rho**3 + (17/16)*c_6*rho**5 -
                                (3/8)*c_7*rho**7 + (1/32)*c_8*rho**9)*eps**5)*np.exp(+1j*np.pi/2)
        E_r_time = np.sum(E_r_spec*trans)*omega_step/(delta_omega*np.sqrt(np.pi))

        B_t_spec = pulse_prep*((c_2*rho)*eps +
                               ((1/2)*c_3*rho + (1/2)*c_4*rho**3 - (1/4)*c_5*rho**5)*eps**3 +
                               ((3/8)*c_4*rho + (3/8)*c_5*rho**3 + (3/16)*c_6*rho**5 -
                                (1/4)*c_7*rho**7 + (1/32)*c_8*rho**9)*eps**5)*np.exp(+1j*np.pi/2)/c
        B_t_time = np.sum(B_t_spec*trans)*omega_step/(delta_omega*np.sqrt(np.pi))

        E_z_total = np.real(Amp*E_z_time)
        E_r_total = np.real(Amp*E_r_time)
        dot_product = v_z[k]*E_z_total + v_r[k]*E_r_total
        B_t_total = np.real(Amp*B_t_time)

        deriv2[k] = (-q_e/(gamma[k]*m_e))*(E_z_total+v_r[k]*B_t_total-v_z[k]*dot_product/(c**2))
        deriv4[k] = (-q_e/(gamma[k]*m_e))*(E_r_total-v_z[k]*B_t_total-v_r[k]*dot_product/(c**2))

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