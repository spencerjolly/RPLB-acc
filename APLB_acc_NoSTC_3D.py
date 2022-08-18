import numpy as np
from numba import jit

@jit(nopython=True)
def APLB_acc_NoSTC_3D(lambda_0, tau_0, w_0, P, Psi_0, phi_2, t_0, z_0, x_0, y_0, beta_0):
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
    t_end = +1400*tau_0
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

    # do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):

        rho = np.sqrt(x[k]**2 + y[k]**2)/w_0
        
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

        B_z_time = -1*pulse_prep*((c_2 - c_3*rho**2)*eps**2 +
                               ((1/2)*c_3 + (1/2)*c_4*rho**2 - (5/4)*c_5*rho**4 + (1/4)*c_6*rho**6)*eps**4)/c

        B_x_time = -1*pulse_prep*((c_2)*eps +
                               (-(1/2)*c_3 + c_4*rho**2 - (1/4)*c_5*rho**4)*eps**3 +
                               (-(3/8)*c_4 - (3/8)*c_5*rho**2 + (17/16)*c_6*rho**4 -
                                (3/8)*c_7*rho**6 + (1/32)*c_8*rho**8)*eps**5)*np.exp(+1j*np.pi/2)*(x[k]/w_0)/c
        
        B_y_time = -1*pulse_prep*((c_2)*eps +
                               (-(1/2)*c_3 + c_4*rho**2 - (1/4)*c_5*rho**4)*eps**3 +
                               (-(3/8)*c_4 - (3/8)*c_5*rho**2 + (17/16)*c_6*rho**4 -
                                (3/8)*c_7*rho**6 + (1/32)*c_8*rho**8)*eps**5)*np.exp(+1j*np.pi/2)*(y[k]/w_0)/c

        E_x_time = pulse_prep*((c_2)*eps +
                               ((1/2)*c_3 + (1/2)*c_4*rho**2 - (1/4)*c_5*rho**4)*eps**3 +
                               ((3/8)*c_4 + (3/8)*c_5*rho**2 + (3/16)*c_6*rho**4 -
                                (1/4)*c_7*rho**6 + (1/32)*c_8*rho**8)*eps**5)*np.exp(+1j*np.pi/2)*(-y[k]/w_0)
        
        E_y_time = pulse_prep*((c_2)*eps +
                               ((1/2)*c_3 + (1/2)*c_4*rho**2 - (1/4)*c_5*rho**4)*eps**3 +
                               ((3/8)*c_4 + (3/8)*c_5*rho**2 + (3/16)*c_6*rho**4 -
                                (1/4)*c_7*rho**6 + (1/32)*c_8*rho**8)*eps**5)*np.exp(+1j*np.pi/2)*(x[k]/w_0)

        B_z_total = np.real(Amp*trans*B_z_time)
        E_x_total = np.real(Amp*trans*E_x_time)
        E_y_total = np.real(Amp*trans*E_y_time)
        dot_product = v_x[k]*E_x_total + v_y[k]*E_y_total
        B_x_total = np.real(Amp*trans*B_x_time)
        B_y_total = np.real(Amp*trans*B_y_time)

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

    KE = (gamma-1)*m_e*c**2/q_e
    return time, z, x, y, v_z, v_x, v_y, KE