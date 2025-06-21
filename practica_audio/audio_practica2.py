import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from IPython.display import Audio, display
import seaborn as sns
import sounddevice as sd
sns.set_context("poster")


# Cargar archivo WAV convertido desde MP3
audio, sr = librosa.load("D:/cosas marco/cosas para beltran/2do año/1er cuatri/procesamiento del habla/practica_audio/AnalisisTextos.wav", sr=None)  # sr=None mantiene la frecuencia original

# Mostrar vector de la señal
indice_inicio = int(0.7 * sr)
print("Vector a partir del segundo 0.7:")
print(audio[indice_inicio:])

# Largo del array
print("Cantidad de elementos de la muestra: ", len(audio))

# Frecuencia de muestreo
print("Frecuencia de Muestreo: ", sr)

# Duración en segundos
duracion = len(audio) / sr
print("Duración (segundos): ", duracion)

# -----------------------------------------------
# PASO 5: IMPRIMIR LA SEÑAL SONORA (FORMA DE ONDA)
# -----------------------------------------------

plt.figure(figsize=(14, 5))  # Un poco más alto
plt.plot(audio, color='dodgerblue', linewidth=1)  # Color más vistoso y línea más fina
plt.title("Forma de onda de la señal", fontsize=18, fontweight='bold')
plt.xlabel("Muestras", fontsize=14)
plt.ylabel("Amplitud", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)  # Agrega grilla suave
plt.tight_layout()  # Ajusta automáticamente los márgenes
plt.show()

# -----------------------------------------------
# REPRODUCIR AUDIO ORIGINAL CON SOUNDDEVICE
# -----------------------------------------------
print("Reproduciendo audio original con sounddevice...")
sd.play(audio, sr)
sd.wait()  # Espera hasta que termine de reproducirse

# Para hacer que el audio dure más (sonido más grave y lento)
sr_mas_lento = sr // 2  # Mitad de la frecuencia original
audio_lento = librosa.resample(audio, orig_sr=sr, target_sr=sr_mas_lento)

# Para hacer que el audio dure menos (sonido más agudo y rápido)
sr_mas_rapido = sr * 2  # Doble de la frecuencia original
audio_rapido = librosa.resample(audio, orig_sr=sr, target_sr=sr_mas_rapido)

# Reproducir más lento (mitad de frecuencia)
print("Reproduciendo más lento...")
sd.play(audio, sr//2)
sd.wait()


# Reproducir más rápido (doble frecuencia)
print("Reproduciendo más rápido...")
sd.play(audio, sr*2)
sd.wait()



# Reducir la frecuencia de muestreo a la mitad (calidad más baja)
nueva_sr = sr // 2  # Mitad de la frecuencia original
audio_baja_calidad = librosa.resample(audio, orig_sr=sr, target_sr=nueva_sr)

# Reproducir el audio de baja calidad
print("Reproduciendo audio con calidad reducida (downsampling)...")
sd.play(audio_baja_calidad, nueva_sr)
sd.wait()



# Reducir la profundidad de bits (de 32/64 bits a 8 bits por ejemplo)
def reducir_profundidad_bits(audio, bits=8):
    """Reduce la profundidad de bits de la señal"""
    max_val = 2**(bits-1)
    return np.round(audio * max_val) / max_val

audio_8bits = reducir_profundidad_bits(audio, bits=8)

# Reproducir audio con menos bits
print("Reproduciendo audio con 8 bits...")
sd.play(audio_8bits, sr)
sd.wait()



# Simular compresión MP3 eliminando componentes frecuenciales
def comprimir_audio(audio, sr, factor=0.5):
    """Elimina parte del contenido frecuencial"""
    D = librosa.stft(audio)  # Transformada de Fourier
    magnitudes = np.abs(D)
    
    # Eliminar una parte de las frecuencias altas
    cutoff = int(magnitudes.shape[0] * factor)
    magnitudes[cutoff:, :] = 0
    
    # Reconstruir la señal
    audio_comprimido = librosa.istft(magnitudes * np.exp(1j * np.angle(D)))
    return audio_comprimido

audio_comprimido = comprimir_audio(audio, sr, factor=0.7)

# Reproducir audio comprimido
print("Reproduciendo audio con compresión de pérdida...")
sd.play(audio_comprimido, sr)
sd.wait()

