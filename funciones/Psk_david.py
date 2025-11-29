import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from funciones.canalisi import generate_canalisi

# =====================
# 1. PARÁMETROS DEL SISTEMA
# =====================
Nbits = 2000
sps = 30
Rb = 10
fs = Rb * sps
fc = 5000
beta = 0.25
span = 30
snr_db = 1000
M = 2  # M-PSK (usa 2 para BPSK, 4 para QPSK, 8, etc.)

# =====================
# FUNCIONES DE MODULACIÓN
# =====================
def bits_to_symbols(bits, M):
    k = int(np.log2(M))
    num_padding = (-len(bits)) % k
    if num_padding > 0:
        bits = np.hstack([bits, np.zeros(num_padding, dtype=int)])
    bits_reshape = bits.reshape((-1, k))
    dec_vals = bits_reshape.dot(1 << np.arange(k)[::-1])
    phases = 2 * np.pi * dec_vals / M
    return np.exp(1j * phases), num_padding

def symbols_to_bits(symbols_rx, M):
    k = int(np.log2(M))
    angles = np.mod(np.angle(symbols_rx), 2*np.pi)
    indices = np.round(angles * M / (2*np.pi)) % M
    bits = (((indices[:, None]).astype(int) >> np.arange(k-1, -1, -1)) & 1).astype(int)
    return bits.reshape(-1)

# =====================
# 2. BITS Y MODULACIÓN
# =====================
bits = np.random.randint(0, 2, Nbits)
symbols, padding = bits_to_symbols(bits, M)

# =====================
# 3. FILTRO RRC
# =====================
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
                (1 - 2/np.pi) * np.cos(np.pi/(4*beta)) )
        else:
            num = (np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta)))
            den = (np.pi*ti*(1-(4*beta*ti)**2))
            h[i] = num / den
    return h / np.sqrt(np.sum(h**2))

rrc = rrc_filter(beta, sps, span)

# =====================
# 4. TRANSMISIÓN
# =====================
upsampled = np.zeros(len(symbols)*sps, dtype=complex)
upsampled[::sps] = symbols
tx_bb = np.convolve(upsampled, rrc, mode='full')

# =====================
# 5. PASABANDA
# =====================
t = np.arange(len(tx_bb)) / fs
tx_pb = np.real(tx_bb * np.exp(1j * 2*np.pi*fc*t))

# =====================
# 6. CANAL
# =====================
rx_pb, h_chan = generate_canalisi(
    tx_signal=tx_pb,
    tipo_fading='Rician',
    nivel_isi='medio',
    snr_db=snr_db,
    max_fase=np.pi
)

# =====================
# 7. DEMODULACIÓN
# =====================
rx_mix = rx_pb * np.exp(-1j * 2*np.pi*fc*t)
rx_bb = np.convolve(rx_mix, rrc, mode='full')

# =====================
# 8. ALINEACIÓN Y CORRECCIÓN
# =====================
delay_rrc = (len(rrc) - 1) // 2
delay_total = 2 * delay_rrc
rx_bb_aligned = rx_bb[delay_total : delay_total + len(upsampled)]
tx_bb_aligned = tx_bb[delay_rrc : delay_rrc + len(upsampled)]

# Corrección de fase
signal_squared = rx_bb_aligned**2
doble_error_fase = np.angle(np.mean(signal_squared))
rx_bb_aligned *= np.exp(-1j * (doble_error_fase / 2))

# Ambigüedad de 180 grados
if np.real(np.vdot(rx_bb_aligned, tx_bb_aligned)) < 0:
    rx_bb_aligned *= -1

# Normalización
rx_bb_aligned /= np.max(np.abs(rx_bb_aligned))
tx_bb_aligned /= np.max(np.abs(tx_bb_aligned))

# =====================
# 9. MUESTREO Y DECISIÓN
# =====================
muestras = rx_bb_aligned[::sps]
bits_rx = symbols_to_bits(muestras, M)
bits_rx = bits_rx[:len(bits)]

# =====================
# 10. BER
# =====================
num_err = np.sum(bits_rx != bits)
ber = num_err / len(bits)
print(f"\nBER = {ber:.3e}   Errores = {num_err}/{len(bits)}\n")

# =====================
# 11. GRÁFICAS
# =====================
plt.figure()
plt.scatter(muestras.real, muestras.imag, alpha=0.5)
plt.title(f"Constelación {M}-PSK")
plt.grid()
plt.axis('equal')
plt.show()