import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- constants ---
K_WATER = 0.57
K_ICE   = 2.2
K_AIR   = 0.025
K_ORG   = 0.25

poros   = 0.9
f_org   = 1.0
f_solid = 1.0 - poros   # 0.1

# solid conductivity (pure organic)
ks    = K_ORG
g_org = 0.33
g_min = g_org * f_org   # 0.33

r = 2.0 / 3.0
f_min = (r * (1 + g_min * (ks/K_WATER - 1))**(-1)
         + (1-r) * (1 + (1 - 2*g_min)*(ks/K_WATER - 1))**(-1))

def devries(wliq, wice):
    wair  = poros - wliq - wice
    g_air = 0.333 * (1.0 - wair / poros)
    g_ice = 0.333 * (1.0 - wice / poros)
    f_gas = (r*(1 + g_air*(K_AIR/K_WATER - 1))**(-1)
             + (1-r)*(1 + (1-2*g_air)*(K_AIR/K_WATER - 1))**(-1))
    f_ice = (r*(1 + g_ice*(K_ICE/K_WATER - 1))**(-1)
             + (1-r)*(1 + (1-2*g_ice)*(K_ICE/K_WATER - 1))**(-1))
    num   = wliq*K_WATER + f_gas*wair*K_AIR + f_min*f_solid*ks + f_ice*wice*K_ICE
    denom = wliq         + f_gas*wair        + f_min*f_solid    + f_ice*wice
    return num / denom

def odonnell(wliq):
    """original O'Donnell (2009), liquid water only"""
    return np.clip(0.032 + 0.5*wliq, 0.04, 0.6)

def odonnell_ice(wliq, wice):
    """additive ice extension"""
    liq_part = np.clip(0.032 + 0.5*wliq, 0.04, 0.6)
    return liq_part + (K_ICE/K_WATER) * 0.5 * wice

def geomean(wliq, wice):
    """geometric mean (JULES-style, previous soil.heat implementation)"""
    wair = poros - wliq - wice
    return (K_ORG**f_solid * K_WATER**wliq * K_ICE**wice * K_AIR**wair)

def porada_ekici(wliq, wice):
    """Porada et al. 2016 / Ekici et al. 2014 organic layer model.
    Kersten-number interpolation between dry (Lo) and a geometric-mean
    saturated conductivity that excludes air (pore-filling components only).
    Lo=0.05 W m-1 K-1 is dry organic conductivity.
    """
    Lo   = 0.05          # dry organic conductivity [W m-1 K-1]
    wtot = wliq + wice
    Ke   = wtot / poros  # Kersten number (degree of saturation)
    # geometric mean over pore-filling components only (no air term)
    L_sat = K_ORG**(1.0 - wtot) * K_WATER**wliq * K_ICE**wice
    return L_sat * Ke + Lo * (1.0 - Ke)

N = 200
theta = np.linspace(0, poros, N)

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.35)

# ---- Panel A: unfrozen ----
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(theta, devries(theta, np.zeros_like(theta)), 'b-', lw=2, label='de Vries')
ax0.plot(theta, odonnell(theta), 'r--', lw=2, label="O'Donnell (liquid only)")
ax0.plot(theta, geomean(theta, np.zeros_like(theta)), 'g:', lw=2, label='Geometric mean')
ax0.plot(theta, porada_ekici(theta, np.zeros_like(theta)), 'm-.', lw=2, label='Porada/Ekici')
ax0.set_xlabel('Volumetric liquid water content (m³ m⁻³)')
ax0.set_ylabel('λ (W m⁻¹ K⁻¹)')
ax0.set_title('A  Unfrozen (w$_{ice}$=0)')
ax0.legend(fontsize=8)
ax0.set_xlim(0, poros)

# ---- Panel B: fully frozen ----
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(theta, devries(np.zeros_like(theta), theta), 'b-', lw=2, label='de Vries')
ax1.plot(theta, odonnell_ice(np.zeros_like(theta), theta), 'r--', lw=2, label="O'Donnell + ice ext.")
ax1.plot(theta, geomean(np.zeros_like(theta), theta), 'g:', lw=2, label='Geometric mean')
ax1.plot(theta, porada_ekici(np.zeros_like(theta), theta), 'm-.', lw=2, label='Porada/Ekici')
ax1.set_xlabel('Volumetric ice content (m³ m⁻³)')
ax1.set_ylabel('λ (W m⁻¹ K⁻¹)')
ax1.set_title('B  Fully frozen (w$_{liq}$=0)')
ax1.legend(fontsize=8)
ax1.set_xlim(0, poros)

# ---- Panel C: fixed total water = 0.7, vary ice fraction ----
ax2 = fig.add_subplot(gs[1, 0])
wtot  = 0.7
ice_f = np.linspace(0, 1, N)
wice_ = ice_f * wtot
wliq_ = (1 - ice_f) * wtot
ax2.plot(ice_f, devries(wliq_, wice_), 'b-', lw=2, label='de Vries')
ax2.plot(ice_f, odonnell_ice(wliq_, wice_), 'r--', lw=2, label="O'Donnell + ice ext.")
ax2.plot(ice_f, geomean(wliq_, wice_), 'g:', lw=2, label='Geometric mean')
ax2.plot(ice_f, porada_ekici(wliq_, wice_), 'm-.', lw=2, label='Porada/Ekici')
ax2.set_xlabel('Ice fraction of total water (-)')
ax2.set_ylabel('λ (W m⁻¹ K⁻¹)')
ax2.set_title(f'C  Fixed total water = {wtot} m³ m⁻³')
ax2.legend(fontsize=8)
ax2.set_xlim(0, 1)

# ---- 2D grid shared by panels D, E, F ----
wliq_2d = np.linspace(0, poros, 120)
wice_2d = np.linspace(0, poros, 120)
WL, WI  = np.meshgrid(wliq_2d, wice_2d)
valid    = (WL + WI) <= poros + 1e-9
DV       = devries(WL, WI)

# ---- Panel F: 2D relative difference – Porada/Ekici vs de Vries ----
ax5 = fig.add_subplot(gs[0, 2])
PE      = porada_ekici(WL, WI)
rel_pe  = np.where(valid, (PE - DV) / DV * 100, np.nan)

cf3 = ax5.contourf(wliq_2d, wice_2d, rel_pe,
                   levels=np.arange(-60, 61, 10), cmap='RdBu_r', extend='both')
ax5.contour(wliq_2d, wice_2d, WL+WI, levels=[poros], colors='k', linewidths=1.5)
ax5.text(0.38, 0.90, 'w$_{liq}$+w$_{ice}$ = porosity', color='k',
         fontsize=7, transform=ax5.transAxes)
ax5.set_xlabel('Volumetric liquid water (m³ m⁻³)')
ax5.set_ylabel('Volumetric ice (m³ m⁻³)')
ax5.set_title('F  (Porada/Ekici − de Vries) / de Vries  [%]')
fig.colorbar(cf3, ax=ax5, label='Relative difference (%)')

# ---- Panel D: 2D relative difference – O'Donnell+ice vs de Vries ----
ax3 = fig.add_subplot(gs[1, 1])
OI       = odonnell_ice(WL, WI)
rel_diff = np.where(valid, (OI - DV) / DV * 100, np.nan)

cf = ax3.contourf(wliq_2d, wice_2d, rel_diff,
                  levels=np.arange(-60, 61, 10), cmap='RdBu_r', extend='both')
ax3.contour(wliq_2d, wice_2d, WL+WI, levels=[poros], colors='k', linewidths=1.5)
ax3.text(0.38, 0.90, 'w$_{liq}$+w$_{ice}$ = porosity', color='k',
         fontsize=7, transform=ax3.transAxes)
ax3.set_xlabel('Volumetric liquid water (m³ m⁻³)')
ax3.set_ylabel('Volumetric ice (m³ m⁻³)')
ax3.set_title("D  (O'Donnell+ice − de Vries) / de Vries  [%]")
fig.colorbar(cf, ax=ax3, label='Relative difference (%)')

# ---- Panel E: 2D relative difference – Geometric mean vs de Vries ----
ax4 = fig.add_subplot(gs[1, 2])
GM       = geomean(WL, WI)
rel_gm   = np.where(valid, (GM - DV) / DV * 100, np.nan)

cf2 = ax4.contourf(wliq_2d, wice_2d, rel_gm,
                   levels=np.arange(-60, 61, 10), cmap='RdBu_r', extend='both')
ax4.contour(wliq_2d, wice_2d, WL+WI, levels=[poros], colors='k', linewidths=1.5)
ax4.text(0.38, 0.90, 'w$_{liq}$+w$_{ice}$ = porosity', color='k',
         fontsize=7, transform=ax4.transAxes)
ax4.set_xlabel('Volumetric liquid water (m³ m⁻³)')
ax4.set_ylabel('Volumetric ice (m³ m⁻³)')
ax4.set_title("E  (Geom. mean − de Vries) / de Vries  [%]")
fig.colorbar(cf2, ax=ax4, label='Relative difference (%)')

for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
    ax.grid(True, alpha=0.3)

fig.suptitle("Thermal conductivity: de Vries vs O'Donnell vs Geometric mean vs Porada/Ekici\n"
             f"Fully organic soil (f_org=1.0, porosity={poros})", fontsize=12)
plt.tight_layout()
plt.savefig('thermal_conductivity_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: thermal_conductivity_comparison.png")
plt.show()
