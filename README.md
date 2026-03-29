# ✨ UniversalVideoCompressor_NVENC ✨

## 🎬 Descripción

**UniversalVideoCompressor_NVENC** es un script avanzado y automatizado para la compresión, escalado y recorte de videos, aprovechando la aceleración por GPU (NVENC) si está disponible. Su objetivo es facilitar el procesamiento de videos de alta calidad con una interfaz interactiva y amigable.

---

### 🚀 Características principales

- 🔍 **Búsqueda automática** de archivos de video en la carpeta del script y subcarpetas.
- 🖥️ **Detección de resolución y colorimetría** usando ffprobe.
- ⚙️ **Opciones de compresión y escalado**: 4K, 1080p, 720p, o mantener resolución original.
- ✂️ **Recorte automático o manual** de barras negras.
- ⚡ **Aceleración por GPU (NVENC)** para decodificación y codificación.
- 🧑‍💻 **Interfaz interactiva** en consola con menús, confirmaciones y paneles enriquecidos (Rich).
- 🎉 **Efectos visuales y sonoros** al finalizar (solo en Windows).
- 🖼️ **Calidad mejorada** en todas las opciones usando codificación `vbr_hq` y ajuste de AQ.

---

## 📦 Requisitos

- Python 3.7 o superior
- [ffmpeg](https://ffmpeg.org/) y [ffprobe](https://ffmpeg.org/) en el PATH del sistema
- GPU NVIDIA compatible con NVENC (opcional, pero recomendado)
- Paquetes Python: `rich`, `Pillow`, `numpy`

Instala las dependencias con:

```sh
pip install rich pillow numpy
```

---

## 🛠️ Uso

1. Coloca el script en la carpeta donde están tus videos.
2. Ejecuta el script desde la terminal:

```sh
python UniversalVideoCompressor_NVENC.py
```

O bien puedes pasar una ruta explícita:

```sh
python UniversalVideoCompressor_NVENC.py "C:\ruta\al\video.mp4"
```

También acepta una carpeta y buscará videos dentro de ella y sus subcarpetas:

```sh
python UniversalVideoCompressor_NVENC.py "C:\ruta\a\mis_videos"
```

3. Sigue las instrucciones en pantalla para:
   - Seleccionar el video a procesar
   - Elegir el tipo de compresión/escalado
   - Configurar el recorte de barras negras (opcional)

---

## ⚙️ Opciones principales

- Compresión **pesada** o **ligera**
- Escalado a **4K**, **1080p**, **720p** o mantener resolución original
- Recorte **automático** (detección de barras negras) o **manual**
- Copia de pistas de **audio** y **subtítulos**

---

## 📝 Notas

- El script está optimizado para **Windows**, pero puede funcionar en otros sistemas (sin efectos visuales/sonoros).
- El archivo de salida se genera junto al original, con sufijos descriptivos para fácil identificación.
- El procesamiento es interactivo y seguro: puedes cancelar en cualquier momento.

---

## 👨‍💻 Autor

**Gilberto Nava Marcos**  
**Versión:** 3.3 estable

---

¡Disfruta de tus videos comprimidos con la mejor calidad y el mínimo esfuerzo!
