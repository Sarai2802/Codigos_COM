import numpy as np
from scipy import signal

def generate_canalisi(tx_signal, tipo_fading='Rayleigh', nivel_isi='bajo', snr_db=20, max_fase=np.pi/8):
    
    def limitar_fase(complejo, max_fase):
        magnitud = np.abs(complejo)
        fase = np.angle(complejo)
        fase_limitada = np.clip(fase, -max_fase, max_fase)
        return magnitud * np.exp(1j * fase_limitada)

    # 1. Configuración de taps 
    if nivel_isi == 'nulo':
        h = np.array([1.0 + 0j])
    else:
        if nivel_isi == 'bajo':
            n_taps = 3; decay_factor = 2.0
        elif nivel_isi == 'medio':
            n_taps = 6; decay_factor = 1.0
        elif nivel_isi == 'alto':
            n_taps = 12; decay_factor = 0.3
        else:
            raise ValueError("Nivel ISI no reconocido")

        pdp = np.exp(-np.arange(n_taps) * decay_factor)
        rayleigh = (np.random.randn(n_taps) + 1j * np.random.randn(n_taps)) * np.sqrt(0.5)
        h = rayleigh * np.sqrt(pdp)

        if tipo_fading == 'Rician':
            K = 10
            los = np.sqrt(K / (K + 1)) + 0j
            nlos = h[0] * np.sqrt(1 / (K + 1))
            h[0] = los + nlos

        # Limitar fase
        h = np.array([limitar_fase(t, max_fase) for t in h])

    # Normalizar energía del canal
    h = h / np.linalg.norm(h)

    # ==========================================================
    # CORRECCIÓN AQUÍ: Evitar el retardo de mode='same'
    # ==========================================================
    # Usamos 'full' y cortamos desde el inicio hasta el largo original.
    # Esto alinea h[0] con tx_signal[0].
    rx_isi = signal.convolve(tx_signal, h, mode='full')[:len(tx_signal)]

    # Ruido AWGN (Igual que antes)
    signal_power = np.mean(np.abs(rx_isi)**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = (np.random.randn(len(rx_isi)) + 1j * np.random.randn(len(rx_isi))) * np.sqrt(noise_power / 2)

    rx = rx_isi + noise

    return rx, h