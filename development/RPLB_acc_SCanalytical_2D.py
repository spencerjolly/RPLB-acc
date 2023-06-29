import numpy as np
from numba import jit

@jit(nopython=True)
def RPLB_acc_SCanalytical_2D(lambda_0, tau_0, w_0, P, Psi_0, t_0, z_0, x_0, beta_0, tau_t):
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
    #perturbation parameter
    b = w_0*tau_t/2
    eps = w_0/z_R
    # amplitude factor
    P_corr = 1 + 3*(eps/2)**2 + 9*(eps/2)**4
    Amp = np.sqrt(8*P/(P_corr*np.pi*e_0*c)) * (omega_0/(2*c))
    
    t_start = t_0 + z_0/c
    t_end = 1e5*tau_0
    # number of time steps per laser period
    n = (lambda_0/(0.8e-6))*200
    num_t = np.int_(np.round(n*(t_end-t_start)/(lambda_0/c)))
    time = np.linspace(t_start, t_end, num_t)
    dt = time[1]-time[0]

    z = np.empty(shape=(len(time)))
    x = np.empty(shape=(len(time)))
    v_z = np.empty(shape=(len(time)))
    v_x = np.empty(shape=(len(time)))
    gamma = np.empty(shape=(len(time)))
    KE = np.zeros(shape=(len(time)))
    deriv2 = np.empty(shape=(len(time)))
    deriv4 = np.empty(shape=(len(time)))

    z[0] = beta_0*c*time[0] + z_0
    x[0] = x_0
    v_z[0] = beta_0*c
    v_x[0] = 0.0
    gamma[0] = 1/np.sqrt(1-beta_0**2)
    KE[0] = ((1/np.sqrt(1-beta_0**2))-1)*m_e*c**2/q_e
    k_stop = -1

    # do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):

        qz = z[k] + 1j*z_R
        alpha = np.sqrt((1/delta_omega)**2 - 1j*omega_0*b**2/(2*c*qz))
        tp = time[k] - z[k]/c
        tpp = tp + omega_0*b*x[k]/(c*qz) - (x[k]**2)*z[k]/(2*c*np.abs(qz)**2)

        phi_G = np.arctan(z[k]/z_R)
        w = w_0*np.sqrt(1+(z[k]/z_R)**2)
        trans = np.exp(-((x[k])/w)**2)

        c_1 = (w_0*np.exp(1j*phi_G)/w)
        c_2 = (w_0*np.exp(1j*phi_G)/w)**2
        c_3 = (w_0*np.exp(1j*phi_G)/w)**3
        c_4 = (w_0*np.exp(1j*phi_G)/w)**4
        c_5 = (w_0*np.exp(1j*phi_G)/w)**5
        c_6 = (w_0*np.exp(1j*phi_G)/w)**6

        gam = (-1j*alpha/b)*(x[k] + 1j*b*tpp/(2*alpha**2))
        con = 1j*b/(2*alpha*w_0)

        H_1 = 2*gam
        H_2 = 4*gam**2 - 2
        H_3 = 8*gam**3 - 12*gam
        H_4 = 16*gam**4 - 48*gam**2 + 12
        H_5 = 32*gam**5 - 160*gam**3 + 120*gam
        H_6 = 64*gam**6 - 480*gam**4 + 720*gam**2 - 120
        H_7 = 128*gam**7 - 1344*gam**5 + 3360*gam**3 - 1680*gam
        H_9 = 256*gam**8 - 3584*gam**6 + 13440*gam**4 - 13440*gam**2 + 1680

        env_temp = np.exp(-(tpp/(2*alpha))**2)
        phase_temp = np.exp(1j*(Psi_0-omega_0*x[k]**2/(2*c*qz)+omega_0*tp))
        pulse_prep = (Amp/(delta_omega*alpha))*c_2*env_temp*phase_temp

        E_z_time = pulse_prep*eps**2*((1 + eps**2*c_1/2) \
                                      - con**2*H_2*(c_1 - eps**2*c_2/2) \
                                      - con**4*H_4*(5*eps**2*c_3/4) \
                                      + con**6*H_6*(eps**2*c_4/4))
        
        E_x_time = 1j*pulse_prep*eps*(con*H_1*(1 - eps**2*c_1/2 - eps**4*3*c_2/8) \
                                      + con**3*H_3*(eps**2*c_2 - eps**4*3*c_3/8) \
                                      - con**5*H_5*(eps**2*c_3/4 - eps**4*17*c_4/16) \
                                      - con**7*H_7*(eps**4*3*c_5/8) \
                                      + con**9*H_9*(eps**4*c_6/32))
        
        B_y_time = 1j*pulse_prep*eps*(con*H_1*(1 + eps**2*c_1/2 + eps**4*3*c_2/8) \
                                      + con**3*H_3*(eps**2*c_2/2 + eps**4*3*c_3/8) \
                                      - con**5*H_5*(eps**2*c_3/4 - eps**4*3*c_4/16) \
                                      - con**7*H_7*(eps**4*c_5/4) \
                                      + con**9*H_9*(eps**4*c_6/32))/c
        
        E_z_total = np.real(E_z_time)
        E_x_total = np.real(E_x_time)
        dot_product = v_z[k]*E_z_total + v_x[k]*E_x_total
        B_y_total = np.real(B_y_time)

        deriv2[k] = (-q_e/(gamma[k]*m_e))*(E_z_total+v_x[k]*B_y_total-v_z[k]*dot_product/(c**2))
        deriv4[k] = (-q_e/(gamma[k]*m_e))*(E_x_total-v_z[k]*B_y_total-v_x[k]*dot_product/(c**2))

        if k==0:
            z[k+1] = z[k] + dt*v_z[k]
            v_z[k+1] = v_z[k] + dt*deriv2[k]
            x[k+1] = x[k] + dt*v_x[k]
            v_x[k+1] = v_x[k] + dt*deriv4[k]
        elif k==1:
            z[k+1] = z[k] + dt*(1.5*v_z[k]-0.5*v_z[k-1])
            v_z[k+1] = v_z[k] + dt*(1.5*deriv2[k]-0.5*deriv2[k-1])
            x[k+1] = x[k] + dt*(1.5*v_x[k]-0.5*v_x[k-1])
            v_x[k+1] = v_x[k] + dt*(1.5*deriv4[k]-0.5*deriv4[k-1])
        elif k==2:
            z[k+1] = z[k] + dt*((23/12)*v_z[k]-(4/3)*v_z[k-1]+(5/12)*v_z[k-2])
            v_z[k+1] = v_z[k] + dt*((23/12)*deriv2[k]-(4/3)*deriv2[k-1]+(5/12)*deriv2[k-2])
            x[k+1] = x[k] + dt*((23/12)*v_x[k]-(4/3)*v_x[k-1]+(5/12)*v_x[k-2])
            v_x[k+1] = v_x[k] + dt*((23/12)*deriv4[k]-(4/3)*deriv4[k-1]+(5/12)*deriv4[k-2])
        elif k==3:
            z[k+1] = z[k] + dt*((55/24)*v_z[k]-(59/24)*v_z[k-1]+(37/24)*v_z[k-2]-(3/8)*v_z[k-3])
            v_z[k+1] = v_z[k] + dt*((55/24)*deriv2[k]-(59/24)*deriv2[k-1]+(37/24)*deriv2[k-2]-(3/8)*deriv2[k-3])
            x[k+1] = x[k] + dt*((55/24)*v_x[k]-(59/24)*v_x[k-1]+(37/24)*v_x[k-2]-(3/8)*v_x[k-3])
            v_x[k+1] = v_x[k] + dt*((55/24)*deriv4[k]-(59/24)*deriv4[k-1]+(37/24)*deriv4[k-2]-(3/8)*deriv4[k-3])
        else:
            z[k+1] = z[k] + dt*((1901/720)*v_z[k]-(1387/360)*v_z[k-1]+(109/30)*v_z[k-2]-(637/360)*v_z[k-3]+(251/720)*v_z[k-4])
            v_z[k+1] = v_z[k] + dt*((1901/720)*deriv2[k]-(1387/360)*deriv2[k-1]+(109/30)*deriv2[k-2]-(637/360)*deriv2[k-3]+(251/720)*deriv2[k-4])
            x[k+1] = x[k] + dt*((1901/720)*v_x[k]-(1387/360)*v_x[k-1]+(109/30)*v_x[k-2]-(637/360)*v_x[k-3]+(251/720)*v_x[k-4])
            v_x[k+1] = v_x[k] + dt*((1901/720)*deriv4[k]-(1387/360)*deriv4[k-1]+(109/30)*deriv4[k-2]-(637/360)*deriv4[k-3]+(251/720)*deriv4[k-4])

        gamma[k+1] = 1/np.sqrt(1-(v_z[k+1]**2+v_x[k+1]**2)/c**2)
        KE[k+1] = (gamma[k+1]-1)*m_e*c**2/q_e
        
        if (time[k] > 300*tau_0 and np.mean(np.abs(np.diff(KE[k-np.int(10*n):k+1]))/(KE[k+1]*dt)) < 1e7):
            k_stop = k+1
            break
            
    return time[:k_stop], z[:k_stop], x[:k_stop], v_z[:k_stop], v_x[:k_stop], KE[:k_stop]