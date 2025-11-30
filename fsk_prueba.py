import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fft, fftfreq, fftshift
# Asegúrate de importar tu función de canal correctamente
from funciones.canalisi import generate_canalisi

# =============================================================================
# 1. PARÁMETROS Y CONFIGURACIÓN
# =============================================================================
if __name__ == "__main__":
    
    # --- CONFIGURACIÓN DE USUARIO ---
    M = 2                  # <--- CAMBIA ESTO (2, 4, 8, 16...)
    Nbits = 4000           # Cantidad de datos
    Rb = 1000              # Tasa de Símbolos (Baudios)
    
    # --- CONTROL DE ESPECTRO ---
    h_index = 2.0          # Índice de modulación.
                           # h=2.0 asegura separación clara de picos.
    
    # --- CONTROL DEL CANAL ---
    snr_db = 25            # SNR alto para visualización clara
    tipo_fading = 'Rician'
    nivel_isi = 'bajo'

    # --- CÁLCULO AUTOMÁTICO DE PARÁMETROS ---
    freq_sep = h_index * Rb 
    
    # Estimamos ancho de banda para definir fs y fc
    max_freq_dev = (M/2) * freq_sep
    
    fc = max_freq_dev * 4       # Portadora segura
    if fc < 5000: fc = 5000     
    
    fs = fc * 8                 # Sobremuestreo x8
    sps = int(fs / Rb)          

    print(f"\n[SISTEMA] M={M}-FSK | h={h_index}")
    print(f"[AUTO] fs={fs/1000:.1f}kHz | fc={fc/1000:.1f}kHz | Sep={freq_sep}Hz | SPS={sps}")

    # =========================================================================
    # 2. GENERACIÓN DE SÍMBOLOS
    # =========================================================================
    k = int(np.log2(M))
    Nbits = Nbits - (Nbits % k)
    num_simbolos = Nbits // k

    simbolos_idx = np.random.randint(0, M, num_simbolos)

    # Mapeo PAM (-3, -1, 1, 3...)
    pam_simbolos = (2 * simbolos_idx) - (M - 1)

    # =========================================================================
    # 3. TRANSMISOR (CPFSK - FASE CONTINUA)
    # =========================================================================
    # 1. Tren de impulsos
    frecuencia_inst = np.repeat(pam_simbolos, sps)
    
    # 2. Integrador de Fase
    phase = np.pi * h_index * np.cumsum(frecuencia_inst) / sps
    
    # 3. Banda Base
    tx_bb = np.exp(1j * phase)

    # =========================================================================
    # 4. CONVERSIÓN A PASABANDA Y CANAL
    # =========================================================================
    t = np.arange(len(tx_bb)) / fs
    carrier = np.exp(1j * 2*np.pi*fc*t)
    tx_pb = np.real(tx_bb * carrier)

    # --- CANAL ---
    rx_pb_raw_complex, h_chan = generate_canalisi(
        tx_pb,
        snr_db=snr_db,
        tipo_fading=tipo_fading,
        nivel_isi=nivel_isi,
        max_fase=np.pi/16 
    )
    rx_pb = np.real(rx_pb_raw_complex)

    # =========================================================================
    # 5. RECEPTOR (DISCRIMINADOR DE FRECUENCIA)
    # =========================================================================
    # 1. Bajar a Banda Base
    rx_bb_raw = signal.hilbert(rx_pb) * np.exp(-1j * 2 * np.pi * fc * t[:len(rx_pb)])

    # 2. Sincronización Temporal
    lags = signal.correlation_lags(len(rx_bb_raw), len(tx_bb), mode='full')
    corr = signal.correlate(rx_bb_raw, tx_bb, mode='full')
    delay = lags[np.argmax(np.abs(corr))]
    if delay < 0: delay = 0

    # Alineación
    L = min(len(tx_bb), len(rx_bb_raw) - delay)
    rx_bb = rx_bb_raw[delay : delay + L]
    tx_bb = tx_bb[:L]

    # 3. Detector de Frecuencia Instantánea
    phase_rx = np.unwrap(np.angle(rx_bb))
    factor_escala = (np.pi * h_index / sps)
    frecuencia_detectada = np.diff(phase_rx) / factor_escala
    frecuencia_detectada = np.append(frecuencia_detectada, 0) 

    # 4. Filtro Post-Detección
    window = np.ones(sps) / sps
    frecuencia_suave = np.convolve(frecuencia_detectada, window, mode='same')

    # 5. Muestreo y Decisión
    muestras = frecuencia_suave[sps//2::sps]
    limit_sym = min(len(muestras), len(simbolos_idx))
    muestras = muestras[:limit_sym]

    niveles_validos = np.arange(-(M-1), M, 2)
    decisiones = []
    for m in muestras:
        idx = (np.abs(niveles_validos - m)).argmin()
        decisiones.append(idx)
    decisiones = np.array(decisiones)

    # =========================================================================
    # 6. RESULTADOS
    # =========================================================================
    errores = np.sum(decisiones != simbolos_idx[:limit_sym])
    ser = errores / limit_sym
    print(f"\n[RESULTADO] SER: {ser:.4f} ({errores}/{limit_sym} errores)")

    # =========================================================================
    # 7. GRÁFICAS (CON CORRECCIÓN DE ESPECTRO)
    # =========================================================================
    
    # G1: Comparación Temporal
    plt.figure(figsize=(10, 5))
    zoom_samples = 6 * sps 
    t_zoom = t[:zoom_samples]
    plt.plot(t_zoom, np.real(tx_bb[:zoom_samples]), label='Tx (Ideal)', linewidth=2)
    plt.plot(t_zoom, np.real(rx_bb[:zoom_samples]) / np.max(np.abs(rx_bb)), '--', label='Rx (Alineada)')
    plt.title(f"Comparación Forma de Onda Banda Base ({M}-FSK)")
    plt.xlabel("Tiempo (s)"); plt.legend(); plt.grid()

    # G2: Señal Pasabanda (Zoom)
    plt.figure(figsize=(10, 4))
    plt.plot(t[:500], rx_pb[:500]) 
    plt.title("Señal Pasabanda Recibida (Zoom)")
    plt.grid()

    # --- G3: ESPECTRO DE POTENCIA (CORREGIDO PARA CNN) ---
    plt.figure(figsize=(10, 5))
    
    # Calcular PSD
    f, Pxx = signal.welch(tx_pb, fs, nperseg=2048)
    f = fftshift(f)
    Pxx = fftshift(Pxx)
    
    # Normalizar a 0 dB máx
    Pxx_db = 10 * np.log10(Pxx + 1e-12)
    Pxx_db_norm = Pxx_db - np.max(Pxx_db)
    
    plt.plot(f, Pxx_db_norm, color='tab:blue')
    
    # RECORTAR FONDO DE RUIDO (LIMPIEZA)
    plt.ylim(bottom=-60, top=2) 
    plt.xlim(0, fs/2) # Solo frecuencias positivas
    
    plt.title(f"Espectro M={M} (Normalizado y Limpio)")
    plt.xlabel("Frecuencia (Hz)"); plt.ylabel("PSD Normalizada (dB)")
    plt.grid(True)

    # G4: Diagrama de Ojo
    plt.figure(figsize=(7, 5))
    L_eye = (len(frecuencia_suave)//(2*sps)) * (2*sps)
    traces = frecuencia_suave[:L_eye].reshape(-1, 2*sps)
    for trace in traces[:100]:
        plt.plot(trace, color='blue', alpha=0.1)
    
    # Niveles teóricos
    for nivel in niveles_validos:
        plt.axhline(nivel, color='red', linestyle='--', alpha=0.5)
        
    plt.title(f"Diagrama de Ojo en Frecuencia (M={M})")
    plt.grid()

    # G5: Respuesta Canal
    plt.figure(figsize=(7,4))
    plt.stem(np.abs(h_chan))
    plt.title("Perfil del Canal (ISI)")
    plt.grid()

    plt.show()