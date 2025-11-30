import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, fftshift
from scipy.signal import convolve
from funciones.canalisi import generate_canalisi

# ==============================
# FUNCIÓN GENERAL M-PSK
# ==============================
def modulate_mpsk(bits, M):
    """
    Devuelve símbolos M-PSK normalizados para cualquier M (2,4,8,16...)
    """
    theta = 2 * np.pi * bits / M

    # Offsets típicos
    if M == 2:
        phase_offset = 0          # BPSK
    elif M == 4:
        phase_offset = np.pi/4    # QPSK Gray
    else:
        phase_offset = 0          # 8PSK, 16PSK...

    symbols = np.exp(1j * (theta + phase_offset))
    return symbols


# ==============================
# 1. PARÁMETROS DEL SISTEMA
# ==============================
M = 2                 # Cambia aquí: 2,4,8,16 para BPSK, QPSK, 8PSK, 16PSK
Nbits = 200
sps = 30
Rb = 1000
fs = Rb * sps
fc = 5000
beta = 0.25
span = 30

# --- CONTROL DEL CANAL 
snr_db = 1000
tipo_fading = 'Rician'
nivel_isi = 'bajo'

# ==============================
# 2. BITS Y MODULACIÓN M-PSK
# ==============================
bits = np.random.randint(0, M, Nbits)
symbols = modulate_mpsk(bits, M)

# ==============================
# 3. FILTRO RRC
# ==============================
def rrc_filter(beta, sps, span):
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + (4*beta/np.pi)
        elif np.isclose(abs(ti), 1/(4*beta)):
            h[i] = (beta/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
                (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            )
        else:
            num = (np.sin(np.pi*ti*(1-beta)) +
                   4*beta*ti*np.cos(np.pi*ti*(1+beta)))
            den = (np.pi*ti*(1-(4*beta*ti)**2))
            h[i] = num / den

    return h / np.sqrt(np.sum(h**2))

rrc = rrc_filter(beta, sps, span)

# ==============================
# 4. SOBREMUESTREO + FILTRADO TX
# ==============================
upsampled = np.zeros(len(symbols) * sps, dtype=complex)
upsampled[::sps] = symbols
tx_bb = np.convolve(upsampled, rrc, mode='full')

# ==============================
# 5. MODULACIÓN PASABANDA
# ==============================
t = np.arange(len(tx_bb)) / fs
carrier = np.exp(1j * 2*np.pi*fc*t)
tx_pb = np.real(tx_bb * carrier)

# ==============================
# 6. CANAL
# ==============================
rx_pb_raw, h_chan = generate_canalisi(
    tx_pb, tipo_fading=tipo_fading,
    nivel_isi=nivel_isi, snr_db=snr_db
)

if len(rx_pb_raw) > len(t):
    rx_pb = rx_pb_raw[:len(t)]
else:
    rx_pb = np.pad(rx_pb_raw, (0, len(t)-len(rx_pb_raw)))

# ==============================
# 7. DEMODULACIÓN COHERENTE
# ==============================
rx_mix = rx_pb * np.exp(-1j * 2*np.pi*fc*t)
rx_bb = np.convolve(rx_mix, rrc, mode='full')

# ==============================
# 8. ALINEACIÓN Y MUESTREO
# ==============================
delay_total = len(rrc) - 1

rx_bb_aligned = rx_bb[delay_total : delay_total + len(upsampled)]

tx_bb_aligned = tx_bb[(len(rrc)-1)//2 : (len(rrc)-1)//2 + len(upsampled)]

muestras_simbolos = rx_bb_aligned[::sps][:Nbits]

# ==============================
# AGC + DEROTACIÓN
# ==============================
amplitud_promedio = np.mean(np.abs(muestras_simbolos))
muestras_simbolos = muestras_simbolos / amplitud_promedio

error_fase = np.angle(np.mean(muestras_simbolos**M)) / M
muestras_simbolos = muestras_simbolos * np.exp(-1j * error_fase)

# ==============================
# DETECCIÓN M-PSK
# ==============================
angles = np.angle(muestras_simbolos) % (2*np.pi)
decisions = np.round(angles * M / (2*np.pi)) % M
decisions = decisions.astype(int)

# ==============================
# 9. BER
# ==============================
bits_rx = decisions[:len(bits)]
num_err = np.sum(bits_rx != bits)
ber = num_err / len(bits_rx)

print(f"\nCanal: {tipo_fading} | ISI: {nivel_isi} | SNR {snr_db}dB")
print(f"BER = {ber:.3e}    Errores = {num_err}/{len(bits_rx)}\n")

# ==============================
# 10. GRÁFICAS
# ==============================

# Señal BB
plt.figure(figsize=(10,5))
plt.plot(np.real(tx_bb_aligned[:800])/np.max(np.abs(tx_bb_aligned)), label='Tx')
plt.plot(np.real(rx_bb_aligned[:800])/np.max(np.abs(rx_bb_aligned)), '--', label='Rx')
plt.title("Señal BB Alineada")
plt.grid(); plt.legend()

# Pasabanda

plt.figure(figsize=(10,4))
delay_visual = (len(rrc) - 1) // 2 

muestras_a_ver = 800
inicio = delay_visual
fin = inicio + muestras_a_ver

# Aseguramos no pasarnos del largo del array
if fin > len(rx_pb): 
    fin = len(rx_pb)

plt.plot(rx_pb[inicio:fin])

plt.title("Señal Recibida en Pasabanda")
plt.xlabel("Muestra"); plt.ylabel("Amplitud")
plt.grid()
# Constelación
plt.figure(figsize=(5,5))
plt.scatter(muestras_simbolos.real, muestras_simbolos.imag, alpha=0.5, label='Recibidos')

# ideales
ideal = modulate_mpsk(np.arange(M), M)
plt.scatter(ideal.real, ideal.imag, color='red', marker='x', s=120, label='Ideal')

plt.axhline(0,color='k'); plt.axvline(0,color='k')
plt.title(f"Constelación M={M}")
plt.grid(); plt.axis('equal'); plt.legend()

# Diagrama de ojo
num_trazas = 50
samples_per_eye = 2*sps
rx_real = np.real(rx_bb_aligned)
num_muestras = (len(rx_real)//samples_per_eye)*samples_per_eye
traces = rx_real[:num_muestras].reshape((-1, samples_per_eye))

plt.figure(figsize=(6,4))
for i in range(min(num_trazas, len(traces))):
    plt.plot(traces[i], alpha=0.3)
plt.title("Diagrama de Ojo")
plt.grid()

# Respuesta canal
plt.figure(figsize=(7,4))
plt.stem(np.abs(h_chan))
plt.title("Respuesta al Impulso del Canal")
plt.grid()

# Espectro
Nfft = 4096
TXF = fftshift(fft(tx_pb[:Nfft]))
RXF = fftshift(fft(rx_pb[:Nfft]))
freqs = fftshift(fftfreq(Nfft, d=1/fs))

plt.figure(figsize=(7,4))
plt.plot(freqs/1000, 20*np.log10(np.abs(TXF)+1e-12), label='Tx')
plt.plot(freqs/1000, 20*np.log10(np.abs(RXF)+1e-12), label='Rx', alpha=0.7)
plt.title("Espectro Pasabanda")
plt.xlabel("Frecuencia [kHz]"); plt.ylabel("Magnitud [dB]")
plt.grid(); plt.legend()

plt.show()
