import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')

# Birim dönüşümü
KM_S_MPC_TO_GYR = 1.022e-3

# Referans modeller
Ho_ref = 67.4      # km/s/Mpc
Om_ref = 0.315     # Madde yoğunluğu
Ol_ref = 0.685     # Karanlık enerji
Or_ref = 9e-5      # Radyasyon
Ok_ref = 0.0       # Eğrilik (ΛCDM)
w_ref  = -1.0      # Karanlık enerji denklem durumu

# No-Lambda modeli
Ol_nolam = 0.0
Ok_nolam = 0.685   # Enerji bütçesini korumak için eğriliğe aktarıldı

LCDM      = dict(Ho=Ho_ref, Om=Om_ref, Ol=Ol_ref,    Or=Or_ref, Ok=Ok_ref,    w=w_ref)
NO_LAMBDA = dict(Ho=Ho_ref, Om=Om_ref, Ol=Ol_nolam,  Or=Or_ref, Ok=Ok_nolam,  w=w_ref)

# Rastgele model aralıkları (Ωλ düşük yanlı)
PARAM_RANGES = {
    "Om": (0.05, 0.95),
    "Ol": (0.00, 0.25),
    "Or": (1e-6, 1e-3),
    "Ok": (-0.5,  0.5),
    "w":  (-2.0,  0.0),
    "Ho": (50.0, 100.0),
}

def random_model(seed=None):
    rng = np.random.default_rng(seed)
    return {k: float(rng.uniform(*v)) for k, v in PARAM_RANGES.items()}

# 1. Friedmann denklemi
def friedmann_eq(a, t, Ho, Om, Ol, Or, Ok, w):
    if a <= 0:
        return 0.0
    Ho_gyr = Ho * KM_S_MPC_TO_GYR
    H2 = Or/a**4 + Om/a**3 + Ok/a**2 + Ol*a**(-3*(1+w))
    dadt = a * Ho_gyr * np.sqrt(max(H2, 0.0))
    return dadt

# 2. Hubble parametresi
def hubble_parameter(z, Ho, Om, Ol, Or, Ok, w):
    return Ho * np.sqrt(
        Or  * (1+z)**4 +
        Om  * (1+z)**3 +
        Ok  * (1+z)**2 +
        Ol  * (1+z)**(3*(1+w))
    )

# 3. İvme (yavaşlama) parametresi
def deceleration_parameter(z, Om, Ol, Or, Ok, w):
    H2 = Or*(1+z)**4 + Om*(1+z)**3 + Ok*(1+z)**2 + Ol*(1+z)**(3*(1+w))
    if H2 <= 0:
        return np.nan
    q = (
        Or  * (1+z)**4           * 1.0 +    # radyasyon: w=1/3 → (1+3w)/2 = 1
        Om  * (1+z)**3           * 0.5 +    # madde:     w=0   → 0.5
        Ol  * (1+z)**(3*(1+w))  * (1+3*w)/2  # karanlık enerji
    ) / H2
    return q

# 4. Enerji yoğunluğu oranları
def energy_fractions(z_arr, Om, Ol, Or, Ok, w):
    zp    = 1 + z_arr
    r     = Or * zp**4
    m     = Om * zp**3
    k     = Ok * zp**2
    l     = Ol * zp**(3*(1+w))
    total = r + m + k + l
    total = np.where(total <= 0, np.nan, total)
    return r/total, m/total, np.abs(k)/total, l/total

# Zaman ve kırmızıya kayma dizileri
t_span = np.linspace(0.01, 30, 3000)   # Gyr
a0     = 1e-4                           # Başlangıç ölçek faktörü (Big Bang sonrası)
z_arr  = np.linspace(0, 5, 500)
z_comp = np.logspace(-1, 4, 800)

def integrate_model(p):
    a_t = odeint(friedmann_eq, a0, t_span,
                 args=(p["Ho"],p["Om"],p["Ol"],p["Or"],p["Ok"],p["w"])).flatten()
    today_idx = np.argmin(np.abs(a_t - 1.0))
    t_today   = t_span[today_idx]
    H_arr = np.array([hubble_parameter(z, **{k:p[k] for k in ("Ho","Om","Ol","Or","Ok","w")}) for z in z_arr])
    q_arr = np.array([deceleration_parameter(z, **{k:p[k] for k in ("Om","Ol","Or","Ok","w")}) for z in z_arr])
    return t_span, a_t, t_today, H_arr, q_arr

# Solidification renkleri
C_RAD  = "#ff6b35"
C_MAT  = "#4ecdc4"
C_CURV = "#ffe66d"
C_DARK = "#a78bfa"

def plot_composition(ax, p, title):
    fr, fm, fk, fl = energy_fractions(z_comp, p["Om"], p["Ol"], p["Or"], p["Ok"], p["w"])
    ax.stackplot(z_comp, fl*100, fm*100, fk*100, fr*100,
                 labels=["Karanlık Enerji", "Madde", "|Eğrilik|", "Radyasyon"],
                 colors=[C_DARK, C_MAT, C_CURV, C_RAD], alpha=0.88)

    z_eq = p["Om"] / max(p["Or"], 1e-10) - 1
    if 0 < z_eq < 1e4:
        ax.axvline(z_eq, color="white", lw=1.2, ls="--", alpha=0.7)
        ax.text(z_eq*1.15, 90, f"z≈{z_eq:.0f}\nmat-rad eş.", color="white",
                fontsize=6, va="top", alpha=0.85)

    if p["Ol"] > 1e-4:
        z_la = (p["Ol"] / max(p["Om"], 1e-10))**(1/3) - 1
        if 0 < z_la < 1e4:
            ax.axvline(z_la, color=C_DARK, lw=1.2, ls=":", alpha=0.9)
            ax.text(z_la*1.15, 68, f"z≈{z_la:.2f}\nmat-Λ eş.", color=C_DARK,
                    fontsize=6, va="top", alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Kırmızıya kayma $z$ (log ölçek)", color="#aaaacc", fontsize=9)
    ax.set_ylabel("Toplam enerji yoğunluğunun %'si", color="#aaaacc", fontsize=9)
    ax.set_title(title, color="white", fontsize=10, pad=6)

    box_txt = (f"Ωm={p['Om']:.3f}  Ωλ={p['Ol']:.3f}\n"
               f"Ωr={p['Or']:.1e}  Ωk={p['Ok']:+.3f}  w={p['w']:.2f}")
    ax.text(0.98, 0.03, box_txt, transform=ax.transAxes,
            fontsize=6.5, color="#ccccee", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#111133", alpha=0.7))
    ax.legend(loc="upper left", fontsize=7, framealpha=0.4,
              facecolor="#111122", labelcolor="white", ncol=2,
              columnspacing=0.8, handlelength=1.2)

# Figure düzeni
fig = plt.figure(figsize=(20, 11))
fig.patch.set_facecolor("#080814")
fig.suptitle("Kozmolojik Model Karşılaştırması – ΛCDM vs No-Λ vs Rastgele Düşük-Λ",
             fontsize=14, fontweight="bold", color="white", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.1],
                       hspace=0.45, wspace=0.32)

ax_a  = fig.add_subplot(gs[0, 0])
ax_H  = fig.add_subplot(gs[0, 1])
ax_q  = fig.add_subplot(gs[0, 2])
ax_c1 = fig.add_subplot(gs[1, 0])
ax_c2 = fig.add_subplot(gs[1, 1])
ax_c3 = fig.add_subplot(gs[1, 2])

BG = "#080814"
for ax in [ax_a, ax_H, ax_q, ax_c1, ax_c2, ax_c3]:
    ax.set_facecolor(BG)
    ax.tick_params(colors="#aaaacc")
    ax.grid(alpha=0.18, color="#334")
    for sp in ax.spines.values():
        sp.set_edgecolor("#334466")

# Üst satır: referans + rastgele modeller
N_RANDOM = 6
SEEDS    = range(N_RANDOM)
cmap     = plt.cm.plasma
colors   = [cmap(i / N_RANDOM) for i in range(N_RANDOM)]

REF_MODELS = [
    (LCDM,      "ΛCDM (Ωλ=0.685)",        "white",   "--", 2.5),
    (NO_LAMBDA, "Karanlık Enerjisiz (Ωλ=0)", "#00ffcc", "-",  2.5),
]

for p_ref, label, color, ls, lw in REF_MODELS:
    t, a, t0, H, q = integrate_model(p_ref)
    ax_a.plot(t,     a, color=color, lw=lw, ls=ls, zorder=10, label=label)
    ax_H.plot(z_arr, H, color=color, lw=lw, ls=ls, zorder=10, label=label)
    ax_q.plot(z_arr, q, color=color, lw=lw, ls=ls, zorder=10, label=label)

_, _, t0_lcdm, _, _ = integrate_model(LCDM)
ax_a.axvline(t0_lcdm, color="white", ls=":", lw=1, alpha=0.4)
ax_a.axhline(1.0,     color="white", ls=":", lw=1, alpha=0.3)
ax_q.axhline(0,       color="white", ls="--", lw=1, alpha=0.5, label="q=0 geçişi")

random_models = []
for i, (seed, color) in enumerate(zip(SEEDS, colors)):
    p = random_model(seed)
    random_models.append(p)
    t, a, t0, H, q = integrate_model(p)
    label = f"R{i+1}  Ωm={p['Om']:.2f} Ωλ={p['Ol']:.2f} Ωk={p['Ok']:+.2f} w={p['w']:.2f}"
    ax_a.plot(t,     a, color=color, lw=1.5, alpha=0.85, label=label)
    ax_H.plot(z_arr, H, color=color, lw=1.5, alpha=0.85, label=label)
    ax_q.plot(z_arr, q, color=color, lw=1.5, alpha=0.85, label=label)
    print(f"[R{i+1}]  Evrenin yaşı={t0:.1f} Gyr  H0={p['Ho']:.1f}  "
          f"Ωm={p['Om']:.3f}  Ωλ={p['Ol']:.3f}  Ωk={p['Ok']:+.3f}  w={p['w']:.3f}")

for ax, xlabel, ylabel, title in [
    (ax_a, "Zaman (Gyr)",       "Ölçek Faktörü $a(t)$",      "Ölçek Faktörü vs Zaman"),
    (ax_H, "Kırmızıya kayma $z$", "$H(z)$ [km/s/Mpc]",        "Hubble Parametresi vs Kırmızıya Kayma"),
    (ax_q, "Kırmızıya kayma $z$", "Yavaşlama Parametresi $q$", "İvmelenme Geçişi"),
]:
    ax.set_xlabel(xlabel, color="#aaaacc", fontsize=9)
    ax.set_ylabel(ylabel, color="#aaaacc", fontsize=9)
    ax.set_title(title, color="white", fontsize=10)
    ax.legend(fontsize=5.8, loc="upper left", framealpha=0.35,
              labelcolor="white", facecolor="#111122")

# Alt satır: Solidification panelleri
plot_composition(ax_c1, LCDM,             "Solidification — ΛCDM (standart)")
plot_composition(ax_c2, NO_LAMBDA,        "Solidification — Karanlık Enerjisiz")
plot_composition(ax_c3, random_models[0],
                 f"Solidification — R1  (Ωλ={random_models[0]['Ol']:.3f})")

fig.text(0.5, 0.465,
         "Her bileşenin evrenin enerji bütçetini nasıl sahiplendiği — ilk ateş topundan bugüne  →",
         ha="center", fontsize=9, color="#7777aa", style="italic")

plt.tight_layout()
plt.savefig("cosmology_random.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nEvrenin bugünkü yaşı (a=1):  {t0_lcdm:.2f} Gyr  (beklenen ~13.8 Gyr)")
print(f"H(z=0) = {hubble_parameter(0, **{k:LCDM[k] for k in ('Ho','Om','Ol','Or','Ok','w')}):.1f} km/s/Mpc  (beklenen ~67.4)")
print(f"q(z=0) = {deceleration_parameter(0, **{k:LCDM[k] for k in ('Om','Ol','Or','Ok','w')}):.3f}  (beklenen ~-0.53)")
print("\nGrafik kaydedildi → cosmology_random.png")