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
import json
import subprocess
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

# —————— SONIDO DE NOTIFICACIÓN (SOLO WINDOWS): SONIDO DE EXPERIENCIA DE MINECRAFT ——————
if sys.platform == "win32":
    import winsound

    def play_minecraft_xp_sound():
        """Reproduce una melodía breve inspirada en el sonido de ganar experiencia."""
        tones = [
            (1760, 70),  # A6
            (2093, 70),  # C7
            (2349, 70),  # D7
            (2637, 140)  # E7
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

        # Intersección (recorte más conservador)
        min_left = max(c[2] for c in crop_results if c)
        min_top = max(c[3] for c in crop_results if c)
        # Para right y bottom, calculamos c[2]+c[0] (left+width) y c[3]+c[1] (top+height)
        max_right = min(c[2] + c[0] for c in crop_results if c)
        max_bottom = min(c[3] + c[1] for c in crop_results if c)

        final_width = max_right - min_left
        final_height = max_bottom - min_top

        for f_path in temp_files:
            if os.path.exists(f_path): os.remove(f_path)

        if final_width <= 0 or final_height <= 0 or final_width > orig_width or final_height > orig_height :
            console.print("[yellow]El recorte detectado no es válido o no es necesario (la imagen ya ocupa todo el cuadro). Se mantendrá sin recorte.[/yellow]")
            return None

        console.print(f"[green]Recorte automático detectado:[/green] Ancho=[bold]{final_width}[/bold], Alto=[bold]{final_height}[/bold], X=[bold]{min_left}[/bold], Y=[bold]{min_top}[/bold]")
        if Confirm.ask(f"¿Aplicar este recorte: {final_width}x{final_height} en ({min_left},{min_top})?", default=True):
            return (final_width, final_height, min_left, min_top)
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

    def normalize(val):
        return (val or "").strip().lower()

    color_prim_n = normalize(color_prim)
    color_trc_n = normalize(color_trc)
    color_space_n = normalize(color_space)

    if ("2020" in color_prim_n or "2020" in color_space_n) and ("2084" in color_trc_n or "b67" in color_trc_n):
        color_params = [
            "-color_primaries", color_prim or "bt2020",
            "-color_trc", color_trc or "smpte2084",
            "-colorspace", color_space or "bt2020nc",
        ]
    elif "709" in color_prim_n or "709" in color_space_n:
        color_params = [
            "-color_primaries", color_prim or "bt709",
            "-color_trc", color_trc or "bt709",
            "-colorspace", color_space or "bt709",
        ]
    elif "601" in color_prim_n or "601" in color_space_n:
        color_params = [
            "-color_primaries", color_prim or "bt601",
            "-color_trc", color_trc or "bt601",
            "-colorspace", color_space or "bt601",
        ]
    else:
        console.print("[yellow]Advertencia: No se pudo detectar el espacio de color. Se usará BT.709 por defecto.[/yellow]")
        color_params = [
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
        ]

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
    # console.print(" ".join(command)) # Descomentar para ver el comando completo
    # Imprimir una versión más legible del comando
    cmd_str_display = []
    temp_str = ""
    for i, part in enumerate(command):
        if part.startswith("-") and temp_str: # Si es una opción y hay algo acumulado
            cmd_str_display.append(temp_str.strip())
            temp_str = part + " "
        elif not part.startswith("-"): # Si es un valor
            temp_str += f"'{part}' " if " " in part else part + " "
        else: # Si es una opción y nada acumulado
            temp_str = part + " "

        if i == len(command) -1 and temp_str: # Último elemento
             cmd_str_display.append(temp_str.strip())

    console.print("[bold]Comando FFmpeg:[/bold]")
    for part_display in cmd_str_display:
        console.print(f"  [dim]{part_display}[/dim]")
    console.print("-" * 30)


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
                show_fireworks_animation()
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


# ---------------------------------------------------------
# 6) MENÚ PRINCIPAL UNIFICADO
# ---------------------------------------------------------
def main():
    console.print(Panel("[bold bright_blue]🎞️ Universal Video Processor HEVC NVENC 🎞️[/bold bright_blue]", expand=False, title_align="center", border_style="blue"))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = None
    exts = [".mkv", ".mp4", ".m2ts", ".ts", ".mov", ".avi", ".webm"] # Ampliado

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

    base_name, _ = os.path.splitext(input_path)
    output_file = base_name + suffix
    
    # Confirmar antes de procesar
    console.print(Panel(f"💾 [bold]Archivo de Salida Propuesto:[/bold] [cyan]{os.path.basename(output_file)}[/cyan]", expand=False, border_style="magenta"))
    if not Confirm.ask(f"¿Proceder con la conversión?", default=True):
        console.print("[yellow]Operación cancelada por el usuario.[/yellow]")
        sys.exit(0)

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