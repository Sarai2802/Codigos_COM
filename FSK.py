import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fft, fftfreq, fftshift
# Importamos tu función de canal
from funciones.canalisi import generate_canalisi

# =============================================================================
# 1. PARÁMETROS DE CONFIGURACIÓN
# =============================================================================
M = 4                  # Orden (2-FSK, 4-FSK, 8-FSK)
h_index = 0.7          # Índice de modulación
usar_gfsk = True       # True=GFSK (Suave), False=CPFSK (Rectangular)
BT = 0.3               # Ancho de banda x Tiempo (Solo GFSK)

# Sistema
Nbits = 2000
sps = 32               
Rb = 1000              
fs = Rb * sps          
fc = 5000              

# Canal
snr_db = 20
tipo_fading = 'Rician'
nivel_isi = 'bajo'

# =============================================================================
# 2. GENERACIÓN DE SÍMBOLOS
# =============================================================================
k = int(np.log2(M))
Nbits = Nbits - (Nbits % k)
num_simbolos = Nbits // k

simbolos_idx = np.random.randint(0, M, num_simbolos)

# Mapeo PAM (Niveles de frecuencia: -3, -1, 1, 3...)
pam_simbolos = (2 * simbolos_idx) - (M - 1)

# =============================================================================
# 3. TRANSMISOR (CPFSK / GFSK)
# =============================================================================
# A. Tren de impulsos
impulsos = np.zeros(len(pam_simbolos) * sps)
impulsos[::sps] = pam_simbolos

# B. Filtro de Frecuencia g(t)
if usar_gfsk:
    # Filtro Gaussiano
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * BT)
    t_gauss = np.linspace(-2, 2, 4 * sps)
    g_t = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-t_gauss**2 / (2 * sigma**2))
    g_t = g_t / np.sum(g_t)
    
    frecuencia_inst = np.convolve(impulsos, g_t, mode='full')
    # Recorte de transitorios del filtro
    start_idx = (len(g_t) - 1) // 2
    frecuencia_inst = frecuencia_inst[start_idx : start_idx + len(impulsos)]
    frecuencia_inst = frecuencia_inst * sps  
    
    delay_filtro = len(g_t) // 2 # Retardo introducido por GFSK
else:
    # Pulso Rectangular (CPFSK)
    g_t = np.ones(sps)
    frecuencia_inst = np.repeat(pam_simbolos, sps)
    delay_filtro = 0

# C. Integrador de Fase (CPFSK)
fase_continua = np.pi * h_index * np.cumsum(frecuencia_inst) / sps

# D. Señal Banda Base Transmitida
tx_bb = np.exp(1j * fase_continua)

# =============================================================================
# 4. PASABANDA (TX)
# =============================================================================
t = np.arange(len(tx_bb)) / fs
carrier = np.exp(1j * 2*np.pi*fc*t)
tx_pb = np.real(tx_bb * carrier)

# =============================================================================
# 5. CANAL + RUIDO
# =============================================================================
# Llamada a tu función
rx_pb_raw, h_chan = generate_canalisi(
    tx_pb, tipo_fading=tipo_fading,
    nivel_isi=nivel_isi, snr_db=snr_db
)

# Ajuste de longitud (igual que en tu código PSK)
if len(rx_pb_raw) > len(t):
    rx_pb = rx_pb_raw[:len(t)]
else:
    rx_pb = np.pad(rx_pb_raw, (0, len(t)-len(rx_pb_raw)))

# =============================================================================
# 6. RECEPTOR: RECUPERACIÓN BANDA BASE
# =============================================================================
# Downconversion ideal (para obtener la envolvente compleja y graficar)
rx_bb_raw = signal.hilbert(rx_pb) * np.exp(-1j * 2 * np.pi * fc * t[:len(rx_pb)])

# =============================================================================
# 7. ALINEACIÓN Y SINCRONIZACIÓN (Para Gráficas y Decisión)
# =============================================================================
# Retardo total = Retardo Filtro Gaussiano + Retardo Canal
delay_canal = len(h_chan) - 1
total_delay = delay_filtro + delay_canal

# Alinear Señales para Gráficas
# Recortamos el inicio "muerto" de Rx y el final excedente
rx_bb_aligned = rx_bb_raw[total_delay:]
# Recortamos el Tx para que empiece en el mismo símbolo
tx_bb_aligned = tx_bb[:len(rx_bb_aligned)]

# Asegurar misma longitud
L_min = min(len(rx_bb_aligned), len(tx_bb_aligned))
rx_bb_aligned = rx_bb_aligned[:L_min]
tx_bb_aligned = tx_bb_aligned[:L_min]

# =============================================================================
# 8. DETECCIÓN (DISCRIMINADOR DE FRECUENCIA)
# =============================================================================
# 1. Derivada de la fase (Frecuencia Instantánea)
fase_rx = np.unwrap(np.angle(rx_bb_aligned))
freq_detectada = np.diff(fase_rx)
freq_detectada = np.append(freq_detectada, 0) # Ajuste longitud

# 2. Filtrado post-detección (Suavizado)
window = np.ones(sps) / sps
freq_suave = np.convolve(freq_detectada, window, mode='same')

# 3. Muestreo (Centro del símbolo)
muestras = freq_suave[sps//2::sps]
# Aseguramos no pasarnos
limit_sym = min(len(muestras), len(simbolos_idx))
muestras = muestras[:limit_sym]

# 4. Decisión (Slicer sobre niveles PAM)
# Factor de escala teórico FSK: (pi * h / sps)
factor_escala = (np.pi * h_index / sps)
muestras_norm = muestras / factor_escala

niveles_validos = np.arange(-(M-1), M, 2)
decisiones = []
for m in muestras_norm:
    idx = (np.abs(niveles_validos - m)).argmin()
    decisiones.append(idx)
decisions = np.array(decisiones)

# =============================================================================
# 9. MÉTRICAS (BER/SER)
# =============================================================================
errores = np.sum(decisions != simbolos_idx[:limit_sym])
ser = errores / limit_sym

modo = "GFSK" if usar_gfsk else "CPFSK"
print(f"\n--- RESULTADOS {modo} (M={M}) ---")
print(f"Canal: {tipo_fading} | ISI: {nivel_isi} | SNR: {snr_db}dB")
print(f"SER: {ser:.4f} ({errores}/{limit_sym})")

# =============================================================================
# 10. GRÁFICAS (LAS QUE PEDISTE)
# =============================================================================

# --- GRÁFICA 1: Señal en Banda Base (Tx vs Rx) ---
plt.figure(figsize=(10,5))
# Graficamos la parte REAL (que es una senoidal de frecuencia variable)
plt.plot(np.real(tx_bb_aligned[:800]), label='Transmitida (BB)', linewidth=2)
# Normalizamos Rx visualmente para comparar la forma de onda
factor_vis = np.max(np.abs(tx_bb_aligned)) / np.max(np.abs(rx_bb_aligned))
plt.plot(np.real(rx_bb_aligned[:800]) * factor_vis, '--', label='Recibida (BB)', linewidth=1.5)
plt.title(f"Comparación Señal Banda Base ({modo})")
plt.xlabel("Muestra"); plt.ylim(-1.5, 1.5)
plt.legend(); plt.grid()

# --- GRÁFICA 2: Señal Pasabanda ---
plt.figure(figsize=(10,4))
plt.plot(rx_pb[:800], color='orange')
plt.title("Señal Recibida en Pasabanda (Con Ruido e ISI)")
plt.xlabel("Muestra"); plt.ylabel("Amplitud")
plt.grid()

# --- GRÁFICA 3: Frecuencia Instantánea (Diagrama de Ojo FSK) ---
# Esta es la equivalente a la constelación para FSK (muestra los niveles)
plt.figure(figsize=(6, 5))
eye_data = freq_suave[: (limit_sym-5)*sps]
traces = eye_data.reshape(-1, sps)
for trace in traces[:100]:
    plt.plot(np.linspace(-0.5, 0.5, sps), trace / factor_escala, color='blue', alpha=0.1)
plt.title(f"Diagrama de Ojo de Frecuencia\n(Niveles PAM esperados: {niveles_validos})")
plt.grid()

# --- GRÁFICA 4: Espectro ---
plt.figure(figsize=(8, 4))
f, Pxx_tx = signal.welch(tx_bb, fs, nperseg=1024, return_onesided=False)
f, Pxx_rx = signal.welch(rx_bb_raw, fs, nperseg=1024, return_onesided=False)
f = fftshift(f)
Pxx_tx = fftshift(Pxx_tx); Pxx_rx = fftshift(Pxx_rx)

plt.semilogy(f, Pxx_tx, label='Tx (Limpia)')
plt.semilogy(f, Pxx_rx, label='Rx (Canal)', alpha=0.7)
plt.title(f"Densidad Espectral de Potencia ({modo})")
plt.xlabel("Frecuencia (Hz)"); plt.grid()
plt.legend()

plt.show()