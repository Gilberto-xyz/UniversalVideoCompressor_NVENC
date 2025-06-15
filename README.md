# UniversalVideoCompressor_NVENC

## Descripción
Este script unifica varias utilidades para la compresión, escalado y recorte de videos de forma automática, utilizando aceleración por GPU (NVENC) si está disponible. Permite:

- Buscar automáticamente archivos de video en la carpeta del script.
- Detectar resolución y colorimetría del video con ffprobe.
- Ofrecer múltiples opciones de compresión y escalado (4K, 1080p, 720p, etc.).
- Recorte automático o manual de barras negras.
- Decodificación y codificación acelerada por GPU (NVENC).
- Interfaz interactiva en consola con menús y confirmaciones.
- Efectos visuales y sonoros al finalizar (solo en Windows).

## Requisitos

- Python 3.7+
- [ffmpeg](https://ffmpeg.org/) y [ffprobe](https://ffmpeg.org/) en el PATH del sistema.
- GPU NVIDIA compatible con NVENC (opcional, pero recomendado).
- Paquetes Python: `rich`, `Pillow`, `numpy`

Instalación de dependencias:
```bash
pip install rich pillow numpy
```

## Uso

Coloca el script en la carpeta donde están tus videos y ejecútalo:

```bash
python UniversalVideoCompressor_NVENC.py
```

Sigue las instrucciones en pantalla para seleccionar el video, el tipo de compresión/escalado y el recorte de barras negras.

## Opciones principales

- Compresión pesada o ligera.
- Escalado a 4K, 1080p, 720p, o mantener resolución.
- Recorte automático (detección de barras negras) o manual.
- Copia de pistas de audio y subtítulos.

## Notas

- El script está optimizado para Windows, pero puede funcionar en otros sistemas (sin efectos visuales/sonoros).
- El archivo de salida se genera junto al original, con sufijos descriptivos.

---

**Autor:** [Gilberto Nava Marcos]  
**Versión:** 3.2 estable
