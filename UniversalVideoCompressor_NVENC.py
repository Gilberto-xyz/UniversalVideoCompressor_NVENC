#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
--------------------------
Script unificado para:
 - Buscar automáticamente un archivo de video en la carpeta.
 - Obtener su resolución con ffprobe.
 - Ofrecer múltiples opciones de compresión y escalado (4K, 1080p, etc.)
 - Opción de recorte de barras negras (automática o manual).
 - Decodificación y codificación en GPU (NVENC) si está disponible.

Versión unificada que combina menús "pesados" y "ligeros",
permitiendo escalas desde 720p a 1080p o 4K, y de 1080p a 4K, etc.
"""

import os
import sys
import argparse
import json
import subprocess
import shutil
import tempfile
import math
from PIL import Image
import numpy as np
import glob
import time # Necesario para el sonido y fuegos artificiales
import random # Necesario para los fuegos artificiales

# --- RICH ---
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.live import Live # Necesario para los fuegos artificiales
# --- FIN RICH ---

console = Console()

def print_pretty_command(command: list[str], header: str = "Comando"):
    """Pretty-print a CLI command separating options for readability."""
    cmd_str_display = []
    temp_str = ""
    for idx, part in enumerate(command):
        if part.startswith("-") and temp_str:
            cmd_str_display.append(temp_str.strip())
            temp_str = part + " "
        elif not part.startswith("-"):
            temp_str += f"'{part}' " if " " in part else part + " "
        else:
            temp_str = part + " "
        if idx == len(command) - 1 and temp_str:
            cmd_str_display.append(temp_str.strip())
    console.print(f"[bold]{header}:[/bold]")
    for segment in cmd_str_display:
        console.print(f"  [dim]{segment}[/dim]")

def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Comprime y escala videos usando NVENC con un flujo interactivo."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Ruta de un archivo de video o de una carpeta con videos.",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path_flag",
        help="Ruta de un archivo de video o de una carpeta con videos.",
    )
    return parser.parse_args()
# —————— SONIDO DE NOTIFICACIÓN (SOLO WINDOWS): SONIDO DE EXPERIENCIA DE MINECRAFT ——————
if sys.platform == "win32":
    import winsound

    def play_minecraft_xp_sound():
        """Reproduce una melodía breve inspirada en el sonido de ganar experiencia."""
        # Intentar reproducir archivo WAV si existe
        wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minecraft_levelup.wav")
        if os.path.isfile(wav_path):
            try:
                winsound.PlaySound(wav_path, winsound.SND_FILENAME)
                return
            except Exception:
                pass # Fallback a beeps si falla

        # Secuencia sintética mejorada (Arpegio Fa# Mayor: F#4, A#4, C#5, F#5, A#5, C#6)
        # Frecuencias aproximadas
        tones = [
            (740, 150),   # F#4
            (932, 150),   # A#4
            (1109, 150),  # C#5
            (1480, 150),  # F#5
            (1865, 150),  # A#5
            (2217, 400)   # C#6 (más largo al final)
        ]
        for freq, duration_ms in tones:
            winsound.Beep(freq, duration_ms)
else:
    def play_minecraft_xp_sound():
        pass  # No hacer nada en otros sistemas

# —————— CLASE AUXILIAR PARA UN FUEGO ARTIFICIAL (SOLO WINDOWS) ——————
class Firework:
    def __init__(self, x: int, patterns: list[list[str]]):
        self.x = x
        self.patterns = patterns
        self.frame = 0
        self.max_frame = len(patterns)
    def is_done(self):
        return self.frame >= self.max_frame
    def current_pattern(self):
        if self.frame < self.max_frame:
            return self.patterns[self.frame]
        return [] # Devuelve lista vacía si el frame está fuera de rango

# —————— SHOW_FIREWORKS (SOLO WINDOWS) ——————
def show_fireworks_animation(duration: float = 3.0, fps: int = 10, max_simultaneous: int = 5):
    if sys.platform != "win32":
        return

    base_patterns = [
        ["    ✨    ", "  ✨ ✨  ", " ✨  ✨ ✨ ", "✨   ✨   ✨", " ✨  ✨ ✨ ", "  ✨ ✨  ", "    ✨    "],
        ["    🌟    ", "  🌟 🌟  ", "🌟     🌟", "  🌟 🌟  ", "    🌟    "]
    ]
    all_patterns = []
    for pat in base_patterns:
        seq = [["  ✨  "], ["  🌟  "], ["  🌟  ", " ✨✨✨ ", "  🌟  "], pat]
        fade1 = [line.replace("✨", "·").replace("🌟", "·") for line in pat]
        fade2 = [line.replace("✨", " ").replace("🌟", " ").replace("·", " ") for line in pat]
        seq.extend([fade1, fade2])
        all_patterns.append(seq)

    width = console.width
    if not all_patterns or not any(all_patterns):
        height = 10
    else:
        max_pattern_height = 0
        for seq in all_patterns:
            for p_seq in seq: # p_seq es un patrón individual (lista de strings)
                if p_seq: # Verificar que el patrón no esté vacío
                    max_pattern_height = max(max_pattern_height, len(p_seq))
        height = max(max_pattern_height, 1)


    active_fw: list[Firework] = []
    frame_time = 1 / fps
    start_time = time.perf_counter()

    with Live(console=console, refresh_per_second=fps, transient=True) as live:
        while (time.perf_counter() - start_time) < duration:
            if len(active_fw) < max_simultaneous and random.random() < 0.3:
                if not all_patterns: continue
                x_pos = random.randint(0, max(0, width - 10))
                pattern_seq = random.choice(all_patterns)
                if not pattern_seq: continue
                active_fw.append(Firework(x_pos, pattern_seq))

            canvas = [[" " for _ in range(width)] for _ in range(height)]
            new_active_fw = []
            for fw in active_fw:
                if not fw.is_done():
                    pat = fw.current_pattern()
                    if pat:
                        for row_idx, line in enumerate(pat):
                            col = min(width - len(line), max(0, fw.x))
                            for j, ch in enumerate(line):
                                if row_idx < height and col + j < width:
                                    canvas[row_idx][col + j] = ch
                    fw.frame += 1
                    if not fw.is_done():
                        new_active_fw.append(fw)
            active_fw = new_active_fw

            text = "\n".join("".join(row) for row in canvas)
            live.update(Panel(text, title="🎉 Celebración 🎉", border_style="magenta", expand=False))
            time.sleep(frame_time)

# ---------------------------------------------------------
# 1) UTILIDAD: OBTENER INFORMACIÓN DEL VIDEO (FFprobe)
# ---------------------------------------------------------
def get_video_resolution(input_file):
    if not os.path.isfile(input_file):
        console.print(f"[bold red]Error: No se encontró el archivo de entrada:[/bold red] {input_file}")
        sys.exit(1)

    probe_cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", input_file]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        data = json.loads(result.stdout)
        width = data["streams"][0]["width"]
        height = data["streams"][0]["height"]
        return (width, height)
    except Exception as e:
        console.print(f"[bold red]Error al obtener resolución de video con ffprobe:[/bold red] {e}")
        sys.exit(1)

def get_video_duration(input_file):
    probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_file]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        return float(result.stdout.strip())
    except Exception as e:
        console.print(f"[yellow]Advertencia al obtener duración:[/yellow] {e}. Usando valor por defecto (1800s).")
        return 1800

def get_video_color_info(input_file):
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=color_primaries,color_transfer,color_space",
        "-of", "json", input_file
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return (
            stream.get("color_primaries", ""),
            stream.get("color_transfer", ""),
            stream.get("color_space", "")
        )
    except Exception as e:
        return ("", "", "")

def determine_color_params(color_prim, color_trc, color_space):
    """Return FFmpeg color parameters and individual values."""
    def normalize(val):
        return (val or "").strip().lower()

    color_prim_n = normalize(color_prim)
    color_trc_n = normalize(color_trc)
    color_space_n = normalize(color_space)

    if ("2020" in color_prim_n or "2020" in color_space_n) and ("2084" in color_trc_n or "b67" in color_trc_n):
        params = [
            "-color_primaries", color_prim or "bt2020",
            "-color_trc", color_trc or "smpte2084",
            "-colorspace", color_space or "bt2020nc",
        ]
    elif "709" in color_prim_n or "709" in color_space_n:
        params = [
            "-color_primaries", color_prim or "bt709",
            "-color_trc", color_trc or "bt709",
            "-colorspace", color_space or "bt709",
        ]
    elif "601" in color_prim_n or "601" in color_space_n:
        params = [
            "-color_primaries", color_prim or "bt601",
            "-color_trc", color_trc or "bt601",
            "-colorspace", color_space or "bt601",
        ]
    else:
        console.print("[yellow]Advertencia: No se pudo detectar el espacio de color. Se usará BT.709 por defecto.[/yellow]")
        params = [
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
        ]

    values = {
        "primaries": params[1],
        "transfer": params[3],
        "matrix": params[5],
    }
    return params, values

def detect_dolby_vision(input_file):
    """Detect Dolby Vision metadata (profile/level) using ffprobe."""
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-print_format", "json", "-show_streams",
        input_file,
    ]
    try:
        result = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="ignore",
        )
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return None
        stream = streams[0]
        side_data = stream.get("side_data_list") or []

        def try_int(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return value

        dv_profile = stream.get("dv_profile")
        dv_level = stream.get("dv_level")
        for entry in side_data:
            if dv_profile is None and entry.get("dv_profile") is not None:
                dv_profile = entry.get("dv_profile")
            if dv_level is None and entry.get("dv_level") is not None:
                dv_level = entry.get("dv_level")

        has_dv_side_data = any(
            "dolby" in (entry.get("side_data_type", "") or "").lower()
            or "dovi" in (entry.get("side_data_type", "") or "").lower()
            for entry in side_data
        )
        if dv_profile is None and not has_dv_side_data:
            return None

        info = {
            "profile": try_int(dv_profile),
            "level": try_int(dv_level),
            "codec": stream.get("codec_name"),
            "has_side_data": has_dv_side_data,
        }

        for key in (
            "dv_version_major",
            "dv_version_minor",
            "rpu_present_flag",
            "el_present_flag",
            "bl_present_flag",
            "dv_bl_signal_compatibility_id",
            "dv_md_compression",
        ):
            for entry in side_data:
                if key in entry:
                    info[key] = entry.get(key)
                    break

        if side_data:
            info["side_data_type"] = side_data[0].get("side_data_type")

        return info
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------
# 2) FUNCIÓN PARA GENERAR SCREENSHOT
# ---------------------------------------------------------
def take_screenshot(input_file, screenshot_file, time_sec=900):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda", "-ss", str(time_sec), "-i", input_file, "-frames:v", "1", "-q:v", "2", "-update", "1", "-y", screenshot_file]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[yellow]Error al generar screenshot (se intentará sin aceleración HW):[/yellow] {e}")
        cmd_no_hw = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(time_sec), "-i", input_file, "-frames:v", "1", "-q:v", "2", "-update", "1", "-y", screenshot_file]
        try:
            subprocess.run(cmd_no_hw, check=True)
            return True
        except subprocess.CalledProcessError as e2:
            console.print(f"[red]Error final al generar screenshot:[/red] {e2}")
            return False


# ---------------------------------------------------------
# 3) DETECCIÓN DE ZONA SIN BORDES NEGROS EN UNA IMAGEN
# ---------------------------------------------------------
def detect_non_black_dimensions_full(image_path, threshold=16):
    try:
        image = Image.open(image_path).convert("L")
    except FileNotFoundError:
        console.print(f"[red]Error: No se encontró el archivo de imagen para recorte:[/red] {image_path}")
        return None # O manejar de otra forma
    image_np = np.array(image)
    img_height, img_width = image_np.shape
    mask = image_np > threshold

    non_black_rows = np.any(mask, axis=1)
    non_black_cols = np.any(mask, axis=0)

    if not np.any(non_black_rows) or not np.any(non_black_cols): # Imagen completamente negra
        console.print("[yellow]Advertencia: La imagen para detectar recorte parece ser completamente negra o no tiene contenido no negro.[/yellow]")
        return (image_np.shape[1], image_np.shape[0], 0, 0) # Devuelve dimensiones originales

    top = np.argmax(non_black_rows)
    bottom = len(non_black_rows) - np.argmax(non_black_rows[::-1])
    left = np.argmax(non_black_cols)
    right = len(non_black_cols) - np.argmax(non_black_cols[::-1])

    width = right - left
    height = bottom - top

    # Si la zona detectada es demasiado pequeña, probablemente es una escena muy oscura.
    total_pixels = img_width * img_height if img_width and img_height else 1
    content_ratio = (width * height) / total_pixels
    min_content_ratio = 0.3
    if content_ratio < min_content_ratio:
        console.print(
            f"[yellow]Se descartó el recorte de {os.path.basename(image_path)} "
            f"porque solo cubre el {content_ratio:.1%} del cuadro (posible escena demasiado oscura).[/yellow]"
        )
        return None

    # console.print(f"[dim]Recorte detectado: w={width}, h={height}, l={left}, t={top}, r={right}, b={bottom}[/dim]")
    return (width, height, left, top)

# ---------------------------------------------------------
# 4) SUBMENÚ PARA MANEJO DE BARRAS NEGRAS (CROP)
# ---------------------------------------------------------
def ask_crop_strategy(input_file, orig_width, orig_height):
    console.print(Panel("[bold cyan]🎬 Opciones de Recorte de Barras Negras 🎬[/bold cyan]", expand=False, border_style="blue"))
    console.print("[bold green]1)[/bold green] [bright_magenta]Auto[/bright_magenta] (analizar 5 screenshots y detectar automáticamente)")
    console.print("[bold green]2)[/bold green] [bright_magenta]Manual[/bright_magenta] (ingresar altura para recorte superior/inferior)")
    console.print("[bold green]3)[/bold green] [bright_magenta]Mantener original[/bright_magenta] (sin recortar)")

    choice = Prompt.ask("Selecciona una opción", choices=["1", "2", "3"], default="3")

    if choice == "1":
        console.print("[cyan]Analizando video para recorte automático...[/cyan]")
        duration = get_video_duration(input_file)
        if duration <= 0 : duration = 1800 # fallback
        times = [int(duration * frac) for frac in [0.1, 0.3, 0.5, 0.7, 0.9]]
        if not times: times = [900] # fallback si duration es muy corto

        temp_files = []
        crop_results = []
        base_dir = os.path.dirname(input_file) or "." # Asegurar que base_dir no sea vacío

        for idx, t in enumerate(times):
            screenshot_path = os.path.join(base_dir, f"temp_screenshot_uvp_{idx}.jpg")
            console.print(f"[dim]Generando screenshot de muestra en t={t}s...[/dim]")
            success = take_screenshot(input_file, screenshot_path, time_sec=t)
            if not success:
                console.print(f"[yellow]No se pudo generar screenshot en t={t}. Se omitirá para el análisis de recorte.[/yellow]")
                continue
            temp_files.append(screenshot_path)
            crop_dim = detect_non_black_dimensions_full(screenshot_path, threshold=16)
            if crop_dim:
                crop_results.append(crop_dim)
            else: # Falló la detección en este screenshot
                 console.print(f"[yellow]No se pudo detectar dimensiones en {screenshot_path}.[/yellow]")


        if not crop_results:
            console.print("[yellow]No se pudo detectar zonas no negras en ninguna muestra. Se mantendrá sin recorte.[/yellow]")
            for f_path in temp_files:
                if os.path.exists(f_path): os.remove(f_path)
            return None

        def consensus_margin(values, axis_name, tolerance_px=4, min_ratio=0.6):
            """Devuelve un margen si la mayoría de capturas coincide, o 0 en caso contrario."""
            cleaned = [max(0, int(round(v))) for v in values]
            positives = [v for v in cleaned if v > tolerance_px]
            if not positives:
                return 0
            required = max(1, math.ceil(len(cleaned) * min_ratio))
            if len(positives) < required:
                console.print(
                    f"[dim yellow]Recorte {axis_name} ignorado: solo {len(positives)}/{len(cleaned)} capturas lo detectaron con claridad.[/dim yellow]"
                )
                return 0
            return int(np.median(positives))

        left_candidates = [c[2] for c in crop_results]
        right_candidates = [orig_width - (c[2] + c[0]) for c in crop_results]
        top_candidates = [c[3] for c in crop_results]
        bottom_candidates = [orig_height - (c[3] + c[1]) for c in crop_results]

        crop_left = consensus_margin(left_candidates, "izquierdo")
        crop_right = consensus_margin(right_candidates, "derecho")
        crop_top = consensus_margin(top_candidates, "superior")
        crop_bottom = consensus_margin(bottom_candidates, "inferior")

        final_width = orig_width - (crop_left + crop_right)
        final_height = orig_height - (crop_top + crop_bottom)

        if all(v == 0 for v in [crop_left, crop_right, crop_top, crop_bottom]):
            console.print("[yellow]Las muestras no coincidieron en barras negras consistentes. Se mantendrá sin recorte.[/yellow]")
            for f_path in temp_files:
                if os.path.exists(f_path): os.remove(f_path)
            return None

        for f_path in temp_files:
            if os.path.exists(f_path): os.remove(f_path)

        if final_width <= 0 or final_height <= 0 or final_width > orig_width or final_height > orig_height :
            console.print("[yellow]El recorte detectado no es válido o no es necesario (la imagen ya ocupa todo el cuadro). Se mantendrá sin recorte.[/yellow]")
            return None

        console.print(f"[green]Recorte automático detectado:[/green] Ancho=[bold]{final_width}[/bold], Alto=[bold]{final_height}[/bold], X=[bold]{crop_left}[/bold], Y=[bold]{crop_top}[/bold]")
        if Confirm.ask(f"¿Aplicar este recorte: {final_width}x{final_height} en ({crop_left},{crop_top})?", default=True):
            return (final_width, final_height, crop_left, crop_top)
        else:
            console.print("[yellow]Recorte automático descartado por el usuario.[/yellow]")
            return None


    elif choice == "2":
        console.print(Panel("[bold cyan]✂️ Recorte Manual (Superior/Inferior) ✂️[/bold cyan]", expand=False, border_style="blue"))
        console.print(f"Resolución original: [bold]{orig_width}x{orig_height}[/bold]")
        console.print("Ingresa la [bold]nueva altura[/bold] deseada después del recorte.")
        console.print("Ej: si es 2160 y quieres quitar 100px arriba y 100px abajo, la nueva altura sería 1960.")
        
        new_height = IntPrompt.ask(
            f"Nueva altura (actual: {orig_height}, debe ser <= {orig_height} y par)",
            default=orig_height,
            validate=lambda x: (x <= orig_height and x > 0 and x % 2 == 0) or "Altura inválida"
        )

        diff = orig_height - new_height
        top_crop = diff // 2
        # Asegurar que el alto recortado y el offset sean pares para ffmpeg
        final_cropped_height = orig_height - (top_crop * 2)

        console.print(f"[green]Recorte manual configurado:[/green] Ancho=[bold]{orig_width}[/bold], Alto=[bold]{final_cropped_height}[/bold], X=[bold]0[/bold], Y=[bold]{top_crop}[/bold]")
        return (orig_width, final_cropped_height, 0, top_crop)

    elif choice == "3":
        console.print("[green]Se mantendrá la resolución original sin recorte.[/green]")
        return None
    return None

# ---------------------------------------------------------
# 5) CONSTRUCCIÓN DE COMANDO FFmpeg
# ---------------------------------------------------------
def build_ffmpeg_command(input_file, output_file, params):
    # Detectar espacio de color
    color_prim, color_trc, color_space = get_video_color_info(input_file)
    color_params, _ = determine_color_params(color_prim, color_trc, color_space)

    # Comando base
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats",
        "-hwaccel", "cuda",
        "-i", input_file,
        # "-map_metadata", "-1",           # <--- Si se desea eliminar metadatos, descomentar
        "-map", "0:v:0", "-map", "0:a?", "-map", "0:s?",
        "-c:v", "hevc_nvenc",
        "-bsf:v", "filter_units=remove_types=62|63",  # <--- Añadido filtro de unidades
        "-preset", "p5",
        "-tune", "hq",
        "-profile:v", "main10", "-pix_fmt", "p010le",
        "-rc:v", "vbr_hq",        # mejor calidad con tamaño similar
        "-multipass", "fullres",  # 2-pass interno
        "-bf", "4",
        "-b_ref_mode", "middle",
    ]

    # Parámetros RC
    if "cq" in params and params["cq"]:
        cmd += ["-cq:v", params["cq"]]
    if "bitrate" in params and params["bitrate"]:
        cmd += ["-b:v", params["bitrate"]]
    if "maxrate" in params and params["maxrate"]:
        cmd += ["-maxrate", params["maxrate"]]
    if "bufsize" in params and params["bufsize"]:
        cmd += ["-bufsize", params["bufsize"]]

    # AQ y lookahead
    # Ajustes de AQ más altos para reducir el pixelado sin aumentar mucho el tamaño
    cmd += ["-spatial-aq", "1", "-aq-strength", "12", "-temporal-aq", "1", "-rc-lookahead", "64"]

    # Filtros (crop/scale)
    filters = []
    if params.get("crop"):
        cw, ch, cx, cy = params["crop"]
        filters.append(f"crop={cw}:{ch}:{cx}:{cy}")
    if params.get("scale"):
        filters.append(f"scale={params['scale']}")
        filters.append("format=p010le")
    if filters:
        cmd += ["-vf", ",".join(filters)]

    # Color y copias de audio/subs
    cmd += color_params + ["-c:a", "copy", "-c:s", "copy", output_file]
    return cmd

def run_ffmpeg(command, input_file_basename):
    console.print(Panel(f"[bold yellow]🚀 Ejecutando FFmpeg para '{input_file_basename}' 🚀[/bold yellow]", expand=False, border_style="yellow"))
    print_pretty_command(command, header="Comando FFmpeg")


    try:
        # Usar Popen para poder potencialmente capturar salida en el futuro si se quiere una barra de progreso
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        with console.status("[bold yellow]Procesando video con FFmpeg...", spinner="dots4") as status:
            for line in iter(process.stdout.readline, ''):
                # Aquí se podría parsear la salida de ffmpeg para una barra de progreso
                # Por ahora, solo actualizamos el status para que se vea que está trabajando
                status.update(f"[bold yellow]Procesando: {line.strip()[:console.width-20]}[/bold yellow]")
            process.wait()

        if process.returncode == 0:
            console.print(Panel("✅ [bold green]Compresión completada con éxito.[/bold green] ✅", expand=False, border_style="green"))
            if sys.platform == "win32":
                play_minecraft_xp_sound()
        else:
            console.print(Panel(f"❌ [bold red]Error al ejecutar FFmpeg (código de salida: {process.returncode}).[/bold red] ❌", expand=False, border_style="red"))
            # stderr podría estar en process.stderr si se redirige por separado
            sys.exit(1)

    except FileNotFoundError:
        console.print("[bold red]Error: FFmpeg no encontrado. Asegúrate de que esté en tu PATH.[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error inesperado al ejecutar FFmpeg:[/bold red] {e}")
        sys.exit(1)


def process_dolby_vision(input_file, output_file, params, dv_info):
    profile = dv_info.get("profile")
    profile_text = str(profile) if profile is not None else "desconocido"
    console.print(Panel(
        f"[bold magenta]Dolby Vision detectado (perfil {profile_text}).[/bold magenta]\n"
        "[white]Se utilizara codificacion por CPU (libx265) con inyeccion de RPU perfil 8.1.[/white]",
        expand=False,
        border_style="magenta",
    ))

    dovi_tool_path = shutil.which("dovi_tool")
    if not dovi_tool_path:
        console.print("[bold red]Se requiere 'dovi_tool' en el PATH para procesar Dolby Vision.[/bold red]")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="uvc_dv_") as tmp_dir:
        rpu_path = os.path.join(tmp_dir, "source.rpu.bin")

        console.print(Panel(
            "[bold cyan]Extrayendo metadatos Dolby Vision (RPU) con dovi_tool...[/bold cyan]",
            expand=False,
            border_style="cyan",
        ))
        extract_cmd = [dovi_tool_path]
        if isinstance(profile, int) and profile in (5, 7):
            console.print("[magenta]Convirtiendo RPU a perfil 8.1 (modo -m 2).[/magenta]")
            extract_cmd += ["-m", "2"]
        extract_cmd += ["extract-rpu", "-i", input_file, "-o", rpu_path]
        print_pretty_command(extract_cmd, header="Comando dovi_tool")
        try:
            subprocess.run(extract_cmd, check=True)
        except FileNotFoundError:
            console.print("[bold red]No se encontro 'dovi_tool'.[/bold red]")
            sys.exit(1)
        except subprocess.CalledProcessError as exc:
            console.print(Panel(
                f"[bold red]Error al extraer RPU con dovi_tool:[/bold red] {exc}",
                expand=False,
                border_style="red",
            ))
            sys.exit(1)

        color_prim, color_trc, color_space = get_video_color_info(input_file)
        color_params, color_values = determine_color_params(color_prim, color_trc, color_space)

        filters = []
        if params.get("crop"):
            cw, ch, cx, cy = params["crop"]
            filters.append(f"crop={cw}:{ch}:{cx}:{cy}")
        if params.get("scale"):
            filters.append(f"scale={params['scale']}")
            filters.append("format=yuv420p10le")
        filter_chain = ",".join(filters) if filters else None

        console.print(Panel(
            "[bold magenta]Codificando base layer HDR10 con libx265...[/bold magenta]",
            expand=False,
            border_style="magenta",
        ))

        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats",
            "-i", input_file,
            "-map", "0:v:0", "-map", "0:a?", "-map", "0:s?",
            "-c:v", "libx265",
            "-pix_fmt", "yuv420p10le",
            "-preset", "slow",
            "-profile:v", "main10",
            "-tag:v", "hvc1",
        ]

        if params.get("cq"):
            ffmpeg_cmd += ["-crf", params["cq"]]
        if params.get("bitrate"):
            ffmpeg_cmd += ["-b:v", params["bitrate"]]
        if params.get("maxrate"):
            ffmpeg_cmd += ["-maxrate", params["maxrate"]]
        if params.get("bufsize"):
            ffmpeg_cmd += ["-bufsize", params["bufsize"]]
        if filter_chain:
            ffmpeg_cmd += ["-vf", filter_chain]

        ffmpeg_cmd += color_params

        x265_params = [
            "repeat-headers=1",
            "aud=1",
            "hrd=1",
            "hdr10-opt=1",
            "chromaloc=2",
            "open-gop=0",
            "keyint=240",
            "min-keyint=1",
            f"dolby-vision-profile=8.1",
            f"dolby-vision-rpu={rpu_path.replace(os.sep, '/')}",
        ]

        primaries = color_values.get("primaries")
        transfer = color_values.get("transfer")
        matrix = color_values.get("matrix")
        if primaries:
            x265_params.append(f"colorprim={primaries}")
        if transfer:
            x265_params.append(f"transfer={transfer}")
        if matrix:
            x265_params.append(f"colormatrix={matrix}")

        ffmpeg_cmd += ["-x265-params", ":".join(x265_params)]
        ffmpeg_cmd += ["-c:a", "copy", "-c:s", "copy", output_file]

        run_ffmpeg(ffmpeg_cmd, os.path.basename(input_file))

# ---------------------------------------------------------
# 6) MENÚ PRINCIPAL UNIFICADO
# ---------------------------------------------------------
def main():
    console.print(Panel("[bold bright_blue]🎞️ Universal Video Processor HEVC NVENC 🎞️[/bold bright_blue]", expand=False, title_align="center", border_style="blue"))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    args = parse_cli_args()
    input_path = None
    exts = [".mkv", ".mp4", ".m2ts", ".ts", ".mov", ".avi", ".webm"] # Ampliado

    cli_input = args.input_path_flag or args.input_path
    if cli_input:
        cli_input = os.path.abspath(cli_input)
        if os.path.isdir(cli_input):
            console.print(f"[cyan]Buscando archivos de video en la carpeta indicada:[/cyan] [dim]{cli_input}[/dim]")
            video_files_found = []
            for root, _, files in os.walk(cli_input):
                for f_name in files:
                    if any(f_name.lower().endswith(ext) for ext in exts):
                        video_files_found.append(os.path.join(root, f_name))
            if not video_files_found:
                console.print("[bold red]No se encontró ningún archivo de video compatible en la carpeta indicada ni en subcarpetas.[/bold red]")
                sys.exit(1)
            if len(video_files_found) == 1:
                input_path = video_files_found[0]
                console.print(f"[green]Archivo de video encontrado automáticamente:[/green] [bold yellow]{os.path.basename(input_path)}[/bold yellow]")
            else:
                console.print("[bold yellow]Múltiples archivos de video encontrados. Por favor, selecciona uno:[/bold yellow]")
                for i, vf in enumerate(video_files_found):
                    console.print(f"[cyan]{i+1})[/cyan] {os.path.basename(vf)} ([dim]{os.path.dirname(vf)}[/dim])")

                file_choice_num = IntPrompt.ask(
                    "Ingresa el número del archivo a procesar",
                    choices=[str(j+1) for j in range(len(video_files_found))],
                    show_choices=False
                )
                input_path = video_files_found[file_choice_num - 1]
                console.print(f"[green]Archivo seleccionado:[/green] [bold yellow]{os.path.basename(input_path)}[/bold yellow]")
        else:
            if not os.path.isfile(cli_input):
                console.print(f"[bold red]No se encontró el archivo indicado:[/bold red] {cli_input}")
                sys.exit(1)
            input_path = cli_input
            console.print(f"[green]Usando archivo indicado por parámetro:[/green] [bold yellow]{os.path.basename(input_path)}[/bold yellow]")

    if input_path is None:
        console.print(f"[cyan]Buscando archivos de video en:[/cyan] [dim]{script_dir}[/dim]")
        video_files_found = []
        for root, _, files in os.walk(script_dir):
            for f_name in files:
                if any(f_name.lower().endswith(ext) for ext in exts):
                    video_files_found.append(os.path.join(root, f_name))
        
        if not video_files_found:
            console.print("[bold red]No se encontró ningún archivo de video compatible en la carpeta del script ni en subcarpetas.[/bold red]")
            sys.exit(1)
        
        if len(video_files_found) == 1:
            input_path = video_files_found[0]
            console.print(f"[green]Archivo de video encontrado automáticamente:[/green] [bold yellow]{os.path.basename(input_path)}[/bold yellow]")
        else:
            console.print("[bold yellow]Múltiples archivos de video encontrados. Por favor, selecciona uno:[/bold yellow]")
            for i, vf in enumerate(video_files_found):
                console.print(f"[cyan]{i+1})[/cyan] {os.path.basename(vf)} ([dim]{os.path.dirname(vf)}[/dim])")
            
            file_choice_num = IntPrompt.ask(
                "Ingresa el número del archivo a procesar",
                choices=[str(j+1) for j in range(len(video_files_found))],
                show_choices=False # Los choices ya se mostraron
            )
            input_path = video_files_found[file_choice_num - 1]
            console.print(f"[green]Archivo seleccionado:[/green] [bold yellow]{os.path.basename(input_path)}[/bold yellow]")


    orig_w, orig_h = get_video_resolution(input_path)
    dv_info = detect_dolby_vision(input_path)
    console.print(Panel(f"📹 [bold]Video Original:[/bold] [magenta]{os.path.basename(input_path)}[/magenta] ([yellow]{orig_w}x{orig_h}[/yellow])", expand=False, border_style="green"))

    crop_data = ask_crop_strategy(input_path, orig_w, orig_h)

    is_uhd_ish = (orig_w >= 3200 or orig_h >= 1800) # 4K o cercano
    is_fhd_ish = (1600 <= orig_w < 3200 or 900 <= orig_h < 1800) # 1080p o cercano
    is_hd_ish = (1000 <= orig_w < 1600 or 600 <= orig_h < 900) # 720p o cercano

    console.print(Panel("[bold cyan]⚙️ Opciones de Compresión / Escalado ⚙️[/bold cyan]", expand=False, border_style="blue"))
    
    options = []
    # Lógica de opciones basada en resolución
    if is_uhd_ish: # Si es 4K o cercano
        options = [
            ("BDRip Pesado (Mantener ~4K, alta calidad)", {"cq": "21", "bitrate": "14M", "maxrate": "17.5M", "bufsize": "24M", "scale": None, "suffix": "_BDRip_Pesado_4K.mkv"}),
            ("BDRip Ligero (Mantener ~4K, comp. alta)", {"cq": "24", "bitrate": "10M", "maxrate": "12M", "bufsize": "16M", "scale": None, "suffix": "_BDRip_Ligero_4K.mkv"}),
            ("Downscale a 1080p (alta calidad)", {"cq": "22", "bitrate": "8M", "maxrate": "10M", "bufsize": "12M", "scale": "1920:-2", "suffix": "_BDRip_1080p.mkv"}),
            ("Downscale a 1080p (ligero)", {"cq": "24", "bitrate": "6M", "maxrate": "7.5M", "bufsize": "10M", "scale": "1920:-2", "suffix": "_Ligero_1080p.mkv"}),
        ]
    elif is_fhd_ish: # Si es 1080p o cercano
        options = [
            ("Upscale a ~4K (Pesado)", {"cq": "21", "bitrate": "14M", "maxrate": "17.5M", "bufsize": "24M", "scale": "3840:-2", "suffix": "_Upscale_Pesado_4K.mkv"}),
            ("Upscale a ~4K (Ligero)", {"cq": "24", "bitrate": "10M", "maxrate": "12M", "bufsize": "16M", "scale": "3840:-2", "suffix": "_Upscale_Ligero_4K.mkv"}),
            ("Mantener ~1080p (Pesado)", {"cq": "17", "bitrate": "14M", "maxrate": "17.5M", "bufsize": "24M", "scale": None, "suffix": "_Mantener_1080p_Pesado.mkv"}),
            ("Mantener ~1080p (Ligero)", {"cq": "24", "bitrate": "6M", "maxrate": "7M", "bufsize": "9M", "scale": None, "suffix": "_Mantener_1080p_Ligero.mkv"}),
            ("Downscale a 720p", {"cq": "24", "bitrate": "4M", "maxrate": "5M", "bufsize": "6M", "scale": "1280:-2", "suffix": "_720p.mkv"}),
        ]
    elif is_hd_ish: # Si es 720p o cercano
         options = [
            ("Upscale a ~1080p (Pesado)", {"cq": "21", "bitrate": "8M", "maxrate": "10M", "bufsize": "12M", "scale": "1920:-2", "suffix": "_Up1080p_Pesado.mkv"}),
            ("Upscale a ~1080p (Ligero)", {"cq": "24", "bitrate": "6M", "maxrate": "7M", "bufsize": "9M", "scale": "1920:-2", "suffix": "_Up1080p_Ligero.mkv"}),
            ("Upscale a ~4K (Pesado, no siempre ideal)", {"cq": "21", "bitrate": "14M", "maxrate": "17.5M", "bufsize": "24M", "scale": "3840:-2", "suffix": "_Up4K_Pesado.mkv"}),
            ("Upscale a ~4K (Ligero)", {"cq": "24", "bitrate": "10M", "maxrate": "12M", "bufsize": "16M", "scale": "3840:-2", "suffix": "_Up4K_Ligero.mkv"}),
            ("Mantener ~720p (Pesado)", {"cq": "21", "bitrate": "5M", "maxrate": "6M", "bufsize": "8M", "scale": None, "suffix": "_Mantener_720p_Pesado.mkv"}),
            ("Mantener ~720p (Ligero)", {"cq": "24", "bitrate": "4M", "maxrate": "5M", "bufsize": "7M", "scale": None, "suffix": "_Mantener_720p_Ligero.mkv"}),
        ]
    else: # Resoluciones menores o no estándar
        options = [
            ("Mantener resolución original (comp. media)", {"cq": "23", "bitrate": "4.5M", "maxrate": "5.5M", "bufsize": "7M", "scale": None, "suffix": "_Original_Medio.mkv"}),
            ("Escalar a 1080p (calidad media)", {"cq": "23", "bitrate": "6.5M", "maxrate": "8.5M", "bufsize": "11M", "scale": "1920:-2", "suffix": "_1080p_Medio.mkv"}),
            ("Escalar a 720p (calidad media)", {"cq": "23", "bitrate": "4.5M", "maxrate": "5.5M", "bufsize": "7M", "scale": "1280:-2", "suffix": "_720p_Medio.mkv"}),
        ]

    for i, (desc, _) in enumerate(options):
        console.print(f"[bold green]{i+1})[/bold green] [bright_magenta]{desc}[/bright_magenta]")

    if not options:
        console.print("[bold red]No se pudieron determinar opciones de compresión para esta resolución. Saliendo.[/bold red]")
        sys.exit(1)
        
    choice_num = IntPrompt.ask(
        "Selecciona una opción",
        choices=[str(j+1) for j in range(len(options))],
        show_choices=False
    )
    selected_option_params = options[choice_num - 1][1]
    
    final_params = {
        "crop": crop_data,
        "scale": selected_option_params.get("scale"),
        "cq": selected_option_params.get("cq"),
        "bitrate": selected_option_params.get("bitrate"),
        "maxrate": selected_option_params.get("maxrate"),
        "bufsize": selected_option_params.get("bufsize")
    }
    suffix = selected_option_params.get("suffix", "_procesado.mkv")

    processing_method = "auto"
    if dv_info:
        profile = dv_info.get("profile", "desconocido")
        console.print(Panel(
            f"[bold magenta]Dolby Vision detectado (perfil {profile}).[/bold magenta]\n"
            "[white]Elige el metodo de procesamiento para generar distintas calidades.[/white]",
            expand=False,
            border_style="magenta",
        ))
        method_options = [
            ("Auto (DV 8.1 por CPU si aplica)", "auto"),
            ("Forzar DV 8.1 (CPU con RPU)", "force_dv"),
            ("Forzar NVENC HDR10 (GPU)", "force_nvenc"),
        ]
        for idx, (desc, _) in enumerate(method_options, start=1):
            console.print(f"[bold cyan]{idx})[/bold cyan] {desc}")
        method_choice = IntPrompt.ask(
            "Selecciona el metodo",
            choices=[str(i) for i in range(1, len(method_options) + 1)],
            show_choices=False,
            default="1",
        )
        processing_method = method_options[int(method_choice) - 1][1]
    else:
        processing_method = "force_nvenc"

    base_name, _ = os.path.splitext(input_path)
    output_file = base_name + suffix
    
    # Confirmar antes de procesar
    console.print(Panel(f"💾 [bold]Archivo de Salida Propuesto:[/bold] [cyan]{os.path.basename(output_file)}[/cyan]", expand=False, border_style="magenta"))
    if not Confirm.ask(f"¿Proceder con la conversión?", default=True):
        console.print("[yellow]Operación cancelada por el usuario.[/yellow]")
        sys.exit(0)

    if processing_method == "force_dv":
        process_dolby_vision(input_path, output_file, final_params, dv_info or {})
    elif processing_method == "force_nvenc":
        cmd = build_ffmpeg_command(input_path, output_file, final_params)
        run_ffmpeg(cmd, os.path.basename(input_path))
    else:
        if dv_info:
            process_dolby_vision(input_path, output_file, final_params, dv_info)
        else:
            cmd = build_ffmpeg_command(input_path, output_file, final_params)
            run_ffmpeg(cmd, os.path.basename(input_path))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print_exception(show_locals=True)
        console.log(f"[bold red]Ocurrió un error inesperado en la ejecución principal:[/bold red] {e}")
    finally:
        console.print("\n[bold blue]Script finalizado.[/bold blue]")
