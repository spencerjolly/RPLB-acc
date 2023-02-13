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
    k_0 = omega_0/c
    # spatial parameters as a function of a
    w_00 = (np.sqrt(2)/k_0)*np.sqrt(np.sqrt(1 + (k_0*a)**2) - 1)  # beam waist
    z_R0 = (1/k_0)*(np.sqrt(1 + (k_0*a)**2) - 1)  # Raylegh range
    # amplitude factor
    Amp = -1*np.sqrt(8*P/(np.pi*e_0*c))*a*c/(2*omega_0)
    
    t_start = t_0 + z_0/c
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
    z_omega = z_R0*tau_p*(omega-omega_0)

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
    k_stop = -1

    #do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):

        Rt = np.sqrt(r[k]**2 + (z[k] - z_omega + 1j*a)**2)
        
        E_z_spec = pulse_prep*(2*1j*Amp*np.exp(-omega*a/c)*np.exp(-1j*omega*z_omega/c)/(Rt)**2)*(np.sin(omega*Rt/c)*((2+(omega*r[k]/c)**2)/Rt - 3*r[k]**2/Rt**3)+np.cos(omega*Rt/c)*(3*omega*r[k]**2/(Rt**2*c)-2*omega/c))
        
        E_z_time = np.sum(E_z_spec*np.exp(1j*omega*time[k]))*omega_step/(delta_omega*np.sqrt(np.pi))
        E_z_total = np.real(np.exp(1j*(Psi_0+np.pi/2))*E_z_time)
        
        E_r_spec = pulse_prep*(2*1j*Amp*np.exp(-omega*a/c))*np.exp(-1j*omega*z_omega/c)*(r[k]*(z[k] - z_omega + 1j*a)/Rt**3)*(np.sin(omega*Rt/c)*(3/Rt**2 - (omega/c)**2) - 3*omega*np.cos(omega*Rt/c)/(Rt*c))
        
        E_r_time = np.sum(E_r_spec*np.exp(1j*omega*time[k]))*omega_step/(delta_omega*np.sqrt(np.pi))
        E_r_total = np.real(np.exp(1j*(Psi_0+np.pi/2))*E_r_time)
        
        B_t_spec = pulse_prep*(2*1j*Amp*np.exp(-omega*a/c))*np.exp(-1j*omega*z_omega/c)*(1j*omega*r[k]/(c*Rt)**2)*(np.sin(omega*Rt/c)/Rt - omega*np.cos(omega*Rt/c)/c)
        
        B_t_time = np.sum(B_t_spec*np.exp(1j*omega*time[k]))*omega_step/(delta_omega*np.sqrt(np.pi))
        B_t_total = np.real(np.exp(1j*(Psi_0+np.pi/2))*B_t_time)
        
        dot_product = v_z[k]*E_z_total + v_r[k]*E_r_total
        
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