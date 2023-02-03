import numpy as np
from numba import jit

@jit(nopython=True)
def APLB_acc_NoSTCApril_3D(lambda_0, s, a, P, Psi_0, t_0, z_0, x_0, y_0, beta_0):
    # initialize constants (SI units)
    c = 2.99792458e8  # speed of light
    m_e = 9.10938356e-31
    q_e = 1.60217662e-19
    e_0 = 8.85418782e-12
    # calculate frequency properties
    omega_0 = 2*np.pi*c/lambda_0
    tau_0 = s*np.sqrt(np.exp(2/(s+1))-1)/omega_0
    # amplitude factor
    Amp = -1*np.sqrt(8*P/(np.pi*e_0*c))*a*c/(2*omega_0)
    
    t_start = t_0 + z_0/c
    t_end = +1e5*tau_0
    n = 1500  # number of time steps per laser period
    num_t = np.int_(np.round(n*(t_end-t_start)/(lambda_0/c)))
    time = np.linspace(t_start, t_end, num_t)
    dt = time[1]-time[0]

    # initialize empty arrays
    z = np.empty(shape=(len(time)))
    x = np.empty(shape=(len(time)))
    y = np.empty(shape=(len(time)))
    v_z = np.empty(shape=(len(time)))
    v_x = np.empty(shape=(len(time)))
    v_y = np.empty(shape=(len(time)))
    gamma = np.empty(shape=(len(time)))
    KE = np.zeros(shape=(len(time)))
    deriv2 = np.empty(shape=(len(time)))
    deriv4 = np.empty(shape=(len(time)))
    deriv6 = np.empty(shape=(len(time)))

    # Set initial conditions
    z[0] = beta_0*c*time[0] + z_0
    x[0] = x_0
    y[0] = y_0
    v_z[0] = beta_0*c
    v_x[0] = 0.0
    v_y[0] = 0.0
    gamma[0] = 1/np.sqrt(1-beta_0**2)
    KE[0] = ((1/np.sqrt(1-beta_0**2))-1)*m_e*c**2/q_e
    k_stop = -1

    # do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):
        
        R_t = np.sqrt(x[k]**2 + y[k]**2 + (z[k]+1j*a)**2)
        t_p = time[k] + R_t/c + 1j*a/c
        t_m = time[k] - R_t/c + 1j*a/c
        f_zero_p = (1-1j*omega_0*t_p/s)**(-(s+1))
        f_zero_m = (1-1j*omega_0*t_m/s)**(-(s+1))
        f_one_p = (s+1)*(1j*omega_0/s)*(1-1j*omega_0*t_p/s)**(-(s+2))
        f_one_m = (s+1)*(1j*omega_0/s)*(1-1j*omega_0*t_m/s)**(-(s+2))
        f_two_p = (s+2)*(s+1)*(1j*omega_0/s)**2 * (1-1j*omega_0*t_p/s)**(-(s+3))
        f_two_m = (s+2)*(s+1)*(1j*omega_0/s)**2 * (1-1j*omega_0*t_m/s)**(-(s+3))
        Gm_zero = f_zero_p - f_zero_m
        Gm_one = f_one_p - f_one_m
        Gp_one = f_one_p + f_one_m
        Gm_two = f_two_p - f_two_m
        Gp_two = f_two_p + f_two_m
        Ct = (z[k]+1j*a)/R_t
        St = np.sqrt(x[k]**2 + y[k]**2)/R_t
        S2t = 2*St*Ct
        CEP_term = np.exp(1j*(Psi_0+np.pi/2))
        
        B_z_total = -1*np.real(CEP_term * (Amp/R_t) * (((3*Ct**2 - 1)/R_t) * (Gm_zero/R_t - Gp_one/c) - (St**2*Gm_two/c**2)))/c
        
        B_x_total = -1*np.real(CEP_term * (3*Amp*S2t/(2*R_t)) * ((Gm_zero/R_t**2) - (Gp_one/(c*R_t)) + (Gm_two/(3*c**2))))*(x[k]/np.sqrt(x[k]**2 + y[k]**2))/c
        B_y_total = -1*np.real(CEP_term * (3*Amp*S2t/(2*R_t)) * ((Gm_zero/R_t**2) - (Gp_one/(c*R_t)) + (Gm_two/(3*c**2))))*(y[k]/np.sqrt(x[k]**2 + y[k]**2))/c
        
        E_x_total = np.real(CEP_term * (Amp*St/(c*R_t)) * ((Gm_one/(c*R_t)) - (Gp_two/c**2)))*c*(-y[k]/np.sqrt(x[k]**2 + y[k]**2))
        E_y_total = np.real(CEP_term * (Amp*St/(c*R_t)) * ((Gm_one/(c*R_t)) - (Gp_two/c**2)))*c*(x[k]/np.sqrt(x[k]**2 + y[k]**2))

        dot_product = v_x[k]*E_x_total + v_y[k]*E_y_total

        deriv2[k] = (-q_e/(gamma[k]*m_e))*(v_x[k]*B_y_total-v_y[k]*B_x_total-v_z[k]*dot_product/(c**2))  # Force in z
        deriv4[k] = (-q_e/(gamma[k]*m_e))*(E_x_total+v_y[k]*B_z_total-v_z[k]*B_y_total-v_x[k]*dot_product/(c**2))  # Force in x
        deriv6[k] = (-q_e/(gamma[k]*m_e))*(E_y_total+v_z[k]*B_x_total-v_x[k]*B_z_total-v_y[k]*dot_product/(c**2))  # Force in y

        if k==0:
            z[k+1] = z[k] + dt*v_z[k]
            v_z[k+1] = v_z[k] + dt*deriv2[k]
            x[k+1] = x[k] + dt*v_x[k]
            v_x[k+1] = v_x[k] + dt*deriv4[k]
            y[k+1] = y[k] + dt*v_y[k]
            v_y[k+1] = v_y[k] + dt*deriv6[k]
        elif k==1:
            z[k+1] = z[k] + dt*(1.5*v_z[k]-0.5*v_z[k-1])
            v_z[k+1] = v_z[k] + dt*(1.5*deriv2[k]-0.5*deriv2[k-1])
            x[k+1] = x[k] + dt*(1.5*v_x[k]-0.5*v_x[k-1])
            v_x[k+1] = v_x[k] + dt*(1.5*deriv4[k]-0.5*deriv4[k-1])
            y[k+1] = y[k] + dt*(1.5*v_y[k]-0.5*v_y[k-1])
            v_y[k+1] = v_y[k] + dt*(1.5*deriv6[k]-0.5*deriv6[k-1])
        elif k==2:
            z[k+1] = z[k] + dt*((23/12)*v_z[k]-(4/3)*v_z[k-1]+(5/12)*v_z[k-2])
            v_z[k+1] = v_z[k] + dt*((23/12)*deriv2[k]-(4/3)*deriv2[k-1]+(5/12)*deriv2[k-2])
            x[k+1] = x[k] + dt*((23/12)*v_x[k]-(4/3)*v_x[k-1]+(5/12)*v_x[k-2])
            v_x[k+1] = v_x[k] + dt*((23/12)*deriv4[k]-(4/3)*deriv4[k-1]+(5/12)*deriv4[k-2])
            y[k+1] = y[k] + dt*((23/12)*v_y[k]-(4/3)*v_y[k-1]+(5/12)*v_y[k-2])
            v_y[k+1] = v_y[k] + dt*((23/12)*deriv6[k]-(4/3)*deriv6[k-1]+(5/12)*deriv6[k-2])
        elif k==3:
            z[k+1] = z[k] + dt*((55/24)*v_z[k]-(59/24)*v_z[k-1]+(37/24)*v_z[k-2]-(3/8)*v_z[k-3])
            v_z[k+1] = v_z[k] + dt*((55/24)*deriv2[k]-(59/24)*deriv2[k-1]+(37/24)*deriv2[k-2]-(3/8)*deriv2[k-3])
            x[k+1] = x[k] + dt*((55/24)*v_x[k]-(59/24)*v_x[k-1]+(37/24)*v_x[k-2]-(3/8)*v_x[k-3])
            v_x[k+1] = v_x[k] + dt*((55/24)*deriv4[k]-(59/24)*deriv4[k-1]+(37/24)*deriv4[k-2]-(3/8)*deriv4[k-3])
            y[k+1] = y[k] + dt*((55/24)*v_y[k]-(59/24)*v_y[k-1]+(37/24)*v_y[k-2]-(3/8)*v_y[k-3])
            v_y[k+1] = v_y[k] + dt*((55/24)*deriv6[k]-(59/24)*deriv6[k-1]+(37/24)*deriv6[k-2]-(3/8)*deriv6[k-3])
        else:
            z[k+1] = z[k] + dt*((1901/720)*v_z[k]-(1387/360)*v_z[k-1]+(109/30)*v_z[k-2]-(637/360)*v_z[k-3]+(251/720)*v_z[k-4])
            v_z[k+1] = v_z[k] + dt*((1901/720)*deriv2[k]-(1387/360)*deriv2[k-1]+(109/30)*deriv2[k-2]-(637/360)*deriv2[k-3]+(251/720)*deriv2[k-4])
            x[k+1] = x[k] + dt*((1901/720)*v_x[k]-(1387/360)*v_x[k-1]+(109/30)*v_x[k-2]-(637/360)*v_x[k-3]+(251/720)*v_x[k-4])
            v_x[k+1] = v_x[k] + dt*((1901/720)*deriv4[k]-(1387/360)*deriv4[k-1]+(109/30)*deriv4[k-2]-(637/360)*deriv4[k-3]+(251/720)*deriv4[k-4])
            y[k+1] = y[k] + dt*((1901/720)*v_y[k]-(1387/360)*v_y[k-1]+(109/30)*v_y[k-2]-(637/360)*v_y[k-3]+(251/720)*v_y[k-4])
            v_y[k+1] = v_y[k] + dt*((1901/720)*deriv6[k]-(1387/360)*deriv6[k-1]+(109/30)*deriv6[k-2]-(637/360)*deriv6[k-3]+(251/720)*deriv6[k-4])

        gamma[k+1] = 1/np.sqrt(1-(v_z[k+1]**2+v_x[k+1]**2+v_y[k+1]**2)/c**2)
        KE[k+1] = (gamma[k+1]-1)*m_e*c**2/q_e

        if (time[k] > 300*tau_0 and np.mean(np.abs(np.diff(KE[k-np.int(10*n):k+1]))/(KE[k+1]*dt)) < 1e7):
            k_stop = k+1
            break

    return time[:k_stop], z[:k_stop], x[:k_stop], y[:k_stop], v_z[:k_stop], v_x[:k_stop], v_y[:k_stop], KE[:k_stop]