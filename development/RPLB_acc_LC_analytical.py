import cython_wrappers
import numpy as np
import scipy.special.cython_special
from numba.core import cgutils
from numba.extending import intrinsic,get_cython_function_address
from numba import jit, njit, types
import ctypes

##################Intrinsics###################
@intrinsic
def val_to_double_ptr(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder,args[0])
        return ptr
    sig = types.CPointer(types.float64)(types.float64)
    return sig, impl

@intrinsic
def double_ptr_to_val(typingctx, data):
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val
    sig = types.float64(types.CPointer(types.float64))
    return sig, impl

double = ctypes.c_double
double_p = ctypes.POINTER(double)

addr = get_cython_function_address("cython_wrappers", "erfc")
functype = ctypes.CFUNCTYPE(None,double,double,
                           double_p,double_p,)
erfc_fn_complex = functype(addr)

@njit("complex128(complex128)")
def erfc_complex(val):
    out_real_p=val_to_double_ptr(0.)
    out_imag_p=val_to_double_ptr(0.)

    erfc_fn_complex(np.real(val), np.imag(val),out_real_p,out_imag_p)

    out_real=double_ptr_to_val(out_real_p)
    out_imag=double_ptr_to_val(out_imag_p)

    return np.complex(out_real + 1.j * out_imag)

@jit(nopython=True)
def RPLB_acc_LC_analytical(lambda_0, tau_0, w_0, P, Psi_0, phi_2, phi_3, t_0, z_0, beta_0, tau_p):
    # initialize constants (SI units)
    c = 2.99792458e8  # speed of light
    m_e = 9.10938356e-31
    q_e = 1.60217662e-19
    e_0 = 8.85418782e-12
    # calculate frequency properties
    omega_0 = 2*np.pi*c/lambda_0
    # calculate Rayleigh range
    z_R = (omega_0*w_0**2)/(2*c)
    # amplitude factor
    Amp = np.sqrt(8*P/(np.pi*e_0*c))
    # stretched pulse duration
    tau = np.sqrt(tau_0**2 + (2*phi_2/tau_0)**2)
    
    t_start = t_0/(1-beta_0) + z_0/c
    t_end = +1e5*tau_0
    # number of time steps per laser period
    n = (lambda_0/(0.8e-6))*np.maximum(50, np.round(np.sqrt(P*tau_0/(tau*w_0**2))/(5e10)))  # empirically chosen resolution based on field strength
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

    alpha = tau_0**2 + 2*1j*phi_2
    b = (tau_p**2)/alpha

    #do 5th order Adams-Bashforth finite difference method
    for k in range(0, len(time)-1):
        t_prime = time[k] - z[k]/c
        a = (1 - 1j*z[k]/z_R) - 2*tau_p*t_prime/alpha
        const = (tau_0/np.sqrt(alpha))*(2*np.exp(1j*omega_0*t_prime)/z_R)/(8*b**(3/2))

        field_temp = np.exp(1j*Psi_0)*(-1*np.sqrt(np.pi)*a*np.exp(a**2/(4*b))*erfc_complex(a/(2*np.sqrt(b))) + 2*np.sqrt(b))
        env_temp = np.exp(-(t_prime**2)/alpha)
        field_total = Amp*field_temp*env_temp*const
        deriv2[k] = (-q_e*np.real(field_total)*((1 - beta[k]**2)**(3/2))/(m_e*c))

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