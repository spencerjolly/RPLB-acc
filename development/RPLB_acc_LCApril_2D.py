import numpy as np
from numba import jit

@jit(nopython=True)
def RPLB_acc_LCApril_2D(lambda_0, s, a, P, Psi_0, phi_2, phi_3, t_0, z_0, r_0, beta_0, tau_p):
    # initialize constants (SI units)
    c = 2.99792458e8  # speed of light
    m_e = 9.10938356e-31
    q_e = 1.60217662e-19
    e_0 = 8.85418782e-12
    # calculate frequency properties
    omega_0 = 2*np.pi*c/lambda_0
    tau_0 = s*np.sqrt(np.exp(2/(s+1))-1)/omega_0
    delta_omega = 2/tau_0
    # amplitude factor
    Amp = -1*np.sqrt(8*P/(np.pi*e_0*c))*a*c/(2*omega_0)
    
    t_start = t_0 + z_0/(c*(1-beta_0))
    t_end = +1e5*tau_0
    # number of time steps per laser period
    n = (lambda_0/(0.8e-6))*200  # np.maximum(50, np.round(np.sqrt(P/(w_0**2))/(5e10)))  # empirically chosen resolution based on field strength
    num_t = np.int_(np.round(n*(t_end-t_start)/(lambda_0/c)))
    time = np.linspace(t_start, t_end, num_t)
    dt = time[1]-time[0]
    
    omega = np.linspace((omega_0-4*delta_omega), (omega_0+4*delta_omega), 300)
    omega_step = omega[1]-omega[0]
    
    pulse_temp = np.exp(-((omega-omega_0)/delta_omega)**2)
    pulse_prep = pulse_temp*np.exp(-1j*((phi_2/2)*(omega-omega_0)**2 + (phi_3/6)*(omega-omega_0)**3))
    k = omega/c
    z_omega = (np.sqrt((k*a)**2 + 1) + k*a)*tau_p*(omega-omega_0)/(2*k)

    # initialize empty arrays
    z = np.empty(shape=(len(time)))
    r = np.empty(shape=(len(time)))
    v_z = np.empty(shape=(len(time)))
    v_r = np.empty(shape=(len(time)))
    gamma = np.empty(shape=(len(time)))
    deriv2 = np.empty(shape=(len(time)))
    deriv4 = np.empty(shape=(len(time)))
    KE = np.zeros(shape=(len(time)))

    # Set initial conditions
    z[0] = beta_0*c*time[0] + z_0
    r[0] = r_0
    v_z[0] = beta_0*c
    v_r[0] = 0.0
    gamma[0] = 1/np.sqrt(1-beta_0**2)
    KE[0] = ((1/np.sqrt(1-beta_0**2))-1)*m_e*c**2/q_e
    i_stop = -1

    #do 5th order Adams-Bashforth finite difference method
    for i in range(0, len(time)-1):

        Rt = np.sqrt(r[i]**2 + (z[i] - z_omega + 1j*a)**2)
        
        E_z_spec = pulse_prep*(2*1j*Amp*np.exp(-k*a)*np.exp(-1j*k*z_omega)/(Rt)**2)*(np.sin(k*Rt)*((2+(k*r[i])**2)/Rt - 3*r[i]**2/Rt**3)+np.cos(k*Rt)*(3*k*r[i]**2/(Rt**2)-2*k))
        
        E_z_time = np.sum(E_z_spec*np.exp(1j*omega*time[i]))*omega_step/(delta_omega*np.sqrt(np.pi))
        E_z_total = np.real(np.exp(1j*(Psi_0+np.pi/2))*E_z_time)
        
        E_r_spec = pulse_prep*(2*1j*Amp*np.exp(-k*a))*np.exp(-1j*k*z_omega)*(r[i]*(z[i] - z_omega + 1j*a)/Rt**3)*(np.sin(k*Rt)*(3/Rt**2 - k**2) - 3*k*np.cos(k*Rt)/Rt)
        
        E_r_time = np.sum(E_r_spec*np.exp(1j*omega*time[i]))*omega_step/(delta_omega*np.sqrt(np.pi))
        E_r_total = np.real(np.exp(1j*(Psi_0+np.pi/2))*E_r_time)
        
        B_t_spec = pulse_prep*(2*1j*Amp*np.exp(-k*a))*np.exp(-1j*k*z_omega)*(1j*k*r[i]/(c*Rt**2))*(np.sin(k*Rt)/Rt - k*np.cos(k*Rt))
        
        B_t_time = np.sum(B_t_spec*np.exp(1j*omega*time[i]))*omega_step/(delta_omega*np.sqrt(np.pi))
        B_t_total = np.real(np.exp(1j*(Psi_0+np.pi/2))*B_t_time)
        
        dot_product = v_z[i]*E_z_total + v_r[i]*E_r_total
        
        deriv2[i] = (-q_e/(gamma[i]*m_e))*(E_z_total+v_r[i]*B_t_total-v_z[i]*dot_product/(c**2))
        deriv4[i] = (-q_e/(gamma[i]*m_e))*(E_r_total-v_z[i]*B_t_total-v_r[i]*dot_product/(c**2))

        if i==0:
            z[i+1] = z[i] + dt*v_z[i]
            v_z[i+1] = v_z[i] + dt*deriv2[i]
            r[i+1] = r[i] + dt*v_r[i]
            v_r[i+1] = v_r[i] + dt*deriv4[i]
        elif i==1:
            z[i+1] = z[i] + dt*(1.5*v_z[i]-0.5*v_z[i-1])
            v_z[i+1] = v_z[i] + dt*(1.5*deriv2[i]-0.5*deriv2[i-1])
            r[i+1] = r[i] + dt*(1.5*v_r[i]-0.5*v_r[i-1])
            v_r[i+1] = v_r[i] + dt*(1.5*deriv4[i]-0.5*deriv4[i-1])
        elif i==2:
            z[i+1] = z[i] + dt*((23/12)*v_z[i]-(4/3)*v_z[i-1]+(5/12)*v_z[i-2])
            v_z[i+1] = v_z[i] + dt*((23/12)*deriv2[i]-(4/3)*deriv2[i-1]+(5/12)*deriv2[i-2])
            r[i+1] = r[i] + dt*((23/12)*v_r[i]-(4/3)*v_r[i-1]+(5/12)*v_r[i-2])
            v_r[i+1] = v_r[i] + dt*((23/12)*deriv4[i]-(4/3)*deriv4[i-1]+(5/12)*deriv4[i-2])
        elif i==3:
            z[i+1] = z[i] + dt*((55/24)*v_z[i]-(59/24)*v_z[i-1]+(37/24)*v_z[i-2]-(3/8)*v_z[i-3])
            v_z[i+1] = v_z[i] + dt*((55/24)*deriv2[i]-(59/24)*deriv2[i-1]+(37/24)*deriv2[i-2]-(3/8)*deriv2[i-3])
            r[i+1] = r[i] + dt*((55/24)*v_r[i]-(59/24)*v_r[i-1]+(37/24)*v_r[i-2]-(3/8)*v_r[i-3])
            v_r[i+1] = v_r[i] + dt*((55/24)*deriv4[i]-(59/24)*deriv4[i-1]+(37/24)*deriv4[i-2]-(3/8)*deriv4[i-3])
        else:
            z[i+1] = z[i] + dt*((1901/720)*v_z[i]-(1387/360)*v_z[i-1]+(109/30)*v_z[i-2]-(637/360)*v_z[i-3]+(251/720)*v_z[i-4])
            v_z[i+1] = v_z[i] + dt*((1901/720)*deriv2[i]-(1387/360)*deriv2[i-1]+(109/30)*deriv2[i-2]-(637/360)*deriv2[i-3]+(251/720)*deriv2[i-4])
            r[i+1] = r[i] + dt*((1901/720)*v_r[i]-(1387/360)*v_r[i-1]+(109/30)*v_r[i-2]-(637/360)*v_r[i-3]+(251/720)*v_r[i-4])
            v_r[i+1] = v_r[i] + dt*((1901/720)*deriv4[i]-(1387/360)*deriv4[i-1]+(109/30)*deriv4[i-2]-(637/360)*deriv4[i-3]+(251/720)*deriv4[i-4])

        gamma[i+1] = 1/np.sqrt(1-(v_z[i+1]**2+v_r[i+1]**2)/c**2)
        KE[i+1] = (gamma[i+1]-1)*m_e*c**2/q_e

        if (time[i] > 300*tau_0 and np.mean(np.abs(np.diff(KE[i-np.int(10*n):i+1]))/(KE[i+1]*dt)) < 1e7):
            i_stop = i+1
            break

    return time[:i_stop], z[:i_stop], r[:i_stop], v_z[:i_stop], v_r[:i_stop], KE[:i_stop]