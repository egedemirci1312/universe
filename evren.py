import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#cosmological parameters
Ho = 67.4e-3   # Hubble parameter
Om = 0.315     # matter 
Ol = 0.685     # dark energy
Or = 9e-5      # radiation
Ok = 0.0       # curvature
w  = -1.0      # dark energy equation of state
G  = 6.67430e-11  # gravitational constant

#1. Friedmann equation
def friedmann_eq(a, t):
    if a <= 0:
        return 0.0
    H2 = (Or / a**4) + (Om / a**3) + (Ok / a**2) + (Ol * a**(-3 * (1 + w)))
    dadt = a * Ho * np.sqrt(np.maximum(H2, 0))
    return dadt

#2. Hubble Parameter
def hubble_parameter(z, Om=Om, Ol=Ol, Or=Or, Ok=Ok, w=w):
    return Ho * np.sqrt(
        Or  * (1 + z)**4 +
        Om  * (1 + z)**3 +
        Ok  * (1 + z)**2 +
        Ol  * (1 + z)**(3 * (1 + w))
    )

#3. Deceleration Parameter
def deceleration_parameter(z, Om=Om, Ol=Ol, Or=Or, w=w):
    H2_norm = (
        Or  * (1 + z)**4 +
        Om  * (1 + z)**3 +
        Ol  * (1 + z)**(3 * (1 + w))
    )
    q = (
        Or  * (1 + z)**4 * 1.0  +   # rad: w=1/3 → (1+3w)/2 = 1
        Om  * (1 + z)**3 * 0.5  +   # mat:     w=0   → 0.5
        Ol  * (1 + z)**(3*(1+w)) * (1 + 3*w) / 2   # de: w=-1 → (1+3w)/2 = -1
    ) / H2_norm
    return q

# mathematical solution
t_span = np.linspace(0.01, 30, 3000)   # Gyr
a0 = 1e-4                               # big bang

a_t = odeint(friedmann_eq, a0, t_span).flatten()

# today mark
today_idx = np.argmin(np.abs(a_t - 1.0))
t_today   = t_span[today_idx]

# redshift
z_arr = np.linspace(0, 5, 500)
H_arr = hubble_parameter(z_arr)

# acceleration
q_arr = deceleration_parameter(z_arr)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Simulating the Accelerating Expansion of the Universe", fontsize=14)


ax1 = axes[0]
ax1.plot(t_span, a_t, color='purple', lw=2.5, label="Scale Factor $a(t)$")
ax1.axvline(x=t_today, color='gray',  linestyle='--', label=f"Today (~{t_today:.1f} Gyr)")
ax1.axhline(y=1,        color='black', linestyle=':',  label="$a = 1$ (today)")
ax1.set_xlabel("Time (Gyr)")
ax1.set_ylabel("Scale Factor $a(t)$")
ax1.set_title("Scale Factor vs Time")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

 
ax2 = axes[1]
ax2.plot(z_arr, H_arr, color='black', lw=2.5, label="$H(z)$")
ax2.axvline(x=0, color='gray', linestyle='--', label="Today ($z=0$)")
ax2.set_xlabel("Redshift $z$")
ax2.set_ylabel("$H(z)$ [km/s/Mpc]")
ax2.set_title("Hubble Parameter vs Redshift")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)


ax3 = axes[2]
ax3.plot(z_arr, q_arr, color='crimson', lw=2.5, label="$q(z)$")
ax3.axhline(y=0,  color='black', linestyle='--', lw=1.5, label="$q = 0$ (transition)")
ax3.axvline(x=0,  color='gray',  linestyle=':',  label="Today ($z=0$)")
ax3.fill_between(z_arr, q_arr, 0,
                 where=(q_arr < 0), alpha=0.15, color='blue',   label="Accelerating ($q<0$)")
ax3.fill_between(z_arr, q_arr, 0,
                 where=(q_arr > 0), alpha=0.15, color='orange', label="Decelerating ($q>0$)")
ax3.set_xlabel("Redshift $z$")
ax3.set_ylabel("Deceleration Parameter $q(z)$")
ax3.set_title("Acceleration Transition")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.show()


