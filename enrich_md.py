from PIL import Image
import io
import base64
import re
from openai import OpenAI
from pathlib import Path
import glob
from urllib.parse import unquote

from google.colab import files
import os
import zipfile
import tempfile
import openai
from PIL import Image
import io
import base64
import re
from openai import OpenAI



# ---------- FUNCIONES AUXILIARES ----------

def resize_image_if_needed(image_path: Path, max_size=(480, 480)):
    """
    Redimensiona la imagen *in place* si supera el tamaño dado.
    """
    with Image.open(image_path) as img:
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            img.save(image_path)

def extract_paragraph_context(md_text, image_markdown, window_paragraphs=1):
    """
    Dado el texto completo del markdown y la línea tipo ![...](ruta),
    extrae el párrafo anterior y el siguiente como contexto (o lo más cercano).
    """
    # Dividir en párrafos manteniendo los separadores dobles de newline
    paragraphs = re.split(r'(\n\s*\n)', md_text)  # esto conserva separadores
    # Recombinar en unidades lógicas para reconocer párrafos reales
    # Construimos lista de bloques que no sean solo separadores
    blocks = []
    for i in range(0, len(paragraphs), 2):
        para = paragraphs[i]
        sep = paragraphs[i+1] if i+1 < len(paragraphs) else ""
        blocks.append(para.strip())
    # Buscar en qué párrafo aparece la referencia de imagen
    target_idx = None
    for idx, para in enumerate(blocks):
        if image_markdown in para:
            target_idx = idx
            break
    if target_idx is None:
        # fallback: devolver los dos primeros párrafos
        return "\n\n".join(p for p in blocks[:2] if p)
    parts = []
    if target_idx - 1 >= 0:
        parts.append(blocks[target_idx - 1])
    parts.append(blocks[target_idx])
    if target_idx + 1 < len(blocks):
        parts.append(blocks[target_idx + 1])
    return "\n\n".join(p for p in parts if p)

def describe_image_with_context(client, image_path: Path, context_text: str, model="gpt-4o-mini", max_tokens=200):
    """
    Llama a la API de OpenAI (Responses) enviando imagen + contexto y devuelve la descripción.
    """

    # 🔁 REDUCIR imagen si es necesario
    resize_image_if_needed(image_path, max_size=(480, 480))

    # Leer y codificar imagen a base64
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "La siguiente imagen aparece en este contexto:\n\n"
        f"{context_text}\n\n"
        "Por favor, da una descripción breve (máximo 50 palabras), clara y útil de lo que se ve en la imagen, "
        "relacionándolo con el contexto. Si algo no es claro o parece ambiguo, menciónalo.\n"
        "Recuerda usar no más de 50 palabras."
    )

    #print(f"Prompt:\n{prompt}")

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                ],
            }
        ],
        max_output_tokens=max_tokens,
    )

    # Preferimos output_text si está disponible (comodidad), sino hay que navegar la estructura.
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()
    # Fallback más explícito (puede variar según versiones internas)
    # Algunas respuestas están en response.output[0].content[...]
    try:
        # Buscar algún campo textual en output
        output_items = response.output or []
        texts = []
        for item in output_items:
            if isinstance(item, dict):
                # en algunos formatos viene en 'content' lista de dicts
                if "content" in item and isinstance(item["content"], list):
                    for c in item["content"]:
                        if c.get("type") in ("output_text",) and "text" in c:
                            texts.append(c["text"])
        if texts:
            return "\n".join(texts).strip()
    except Exception:
        pass
    # Si todo falla, devolver repr de la respuesta corta
    return str(response)

def extract_image_descriptions(client, temp_dir):

  # ---------- EJEMPLO DE USO SOBRE .md y sus imágenes ----------
  md_paths = glob.glob(os.path.join(temp_dir, '**', '*.md'), recursive=True)
  md_paths = [Path(p) for p in md_paths]

  # patrón de imagen Markdown: ![alt](ruta)
  #img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
  
  img_pattern = re.compile(r'!\[.*?\]\((.*?\.\w{3,4})\)')

  # Suponiendo que ya tenés md_paths (lista de Path) desde el paso anterior
  # y que las imágenes están referenciadas con rutas relativas en el markdown
  image_descriptions = {}  # mapa rel_path -> descripción

  for md_path in md_paths:  # md_paths debe venir de la etapa previa: list of .md Path objects
      md_text = md_path.read_text(encoding="utf-8")
      matches = img_pattern.findall(md_text)
      for rel_img_ in matches:
          rel_img = unquote(rel_img_)  # <-- convierte %20 a espacio y otros caracteres válidos
          # Normalizar ruta relativa desde el markdown
          abs_img_path = (md_path.parent / rel_img).resolve()
          key = os.path.normpath(os.path.join(os.path.relpath(md_path.parent, start=temp_dir), rel_img))
          if not abs_img_path.is_file():
              print(f"Advertencia: imagen no encontrada")
              continue
          if key in image_descriptions:
              continue  # ya procesada
          #context = extract_paragraph_context(md_text, f'![', window_paragraphs=2 )  # se puede afinar si la imagen tiene alt distinto
          # mejor pasar el fragmento exacto incluyendo la línea de imagen
          # recomponer el markdown snippet con la imagen para contexto de búsqueda
          image_markdown_full = f"![~]({rel_img})"
          context = extract_paragraph_context(md_text, f"({rel_img_})", window_paragraphs=2)  # si usás alt, podés ajustar regex

          # print(f"{rel_img}: {context}")


          desc = describe_image_with_context(client, abs_img_path, context)
          image_descriptions[key] = desc
          #print(f"Procesada imagen {rel_img}: {desc}")

  return md_text, image_descriptions

import re
from pathlib import Path

import re
import base64
import os
from pathlib import Path

def extrae_imagenes_base64(path_md):
    """
    Procesa un archivo Markdown con imágenes en base64 en formato tipo:
    [image1]: <data:image/png;base64,...>

    Extrae las imágenes a una carpeta local (junto al archivo) y reemplaza la sintaxis
    por ![texto](ruta/imagen.png)
    """
    # Asegurar que el archivo existe
    path_md = Path(path_md)
    if not path_md.exists():
        raise FileNotFoundError(f"El archivo {path_md} no existe, baka.")

    # Leer el contenido del archivo
    markdown_text = path_md.read_text(encoding="utf-8")

    # Crear carpeta de destino: misma ruta + carpeta 'imagenes'
    carpeta_destino = path_md.parent / "imagenes"
    carpeta_destino.mkdir(parents=True, exist_ok=True)

    # Regex para detectar imágenes base64
    pattern = re.compile(
        r'\[(?P<id>[^\]]+)\]:\s*<(?P<type>data:image\/[a-zA-Z]+);base64,(?P<data>[A-Za-z0-9+/=]+)>'
    )

    nuevas_lineas = []
    reemplazos = {}
    contador = 0

    for linea in markdown_text.splitlines():
        match = pattern.match(linea)
        if match:
            ext = match.group("type").split("/")[-1]
            b64_data = match.group("data")
            img_bytes = base64.b64decode(b64_data)

            filename = f"imagen_{contador}.{ext}"
            ruta_img_relativa = Path("imagenes") / filename
            ruta_img_absoluta = carpeta_destino / filename

            with open(ruta_img_absoluta, "wb") as f:
                f.write(img_bytes)

            reemplazos[match.group("id")] = ruta_img_relativa.as_posix()
            contador += 1
        else:
            nuevas_lineas.append(linea)

    if contador == 0: return None

    # Reconstruir markdown sin las líneas base64
    texto_actualizado = "\n".join(nuevas_lineas)

    # Reemplazar referencias tipo ![desc][image1]
    for key, ruta in reemplazos.items():
        ref_pattern = re.compile(r'!\[([^\]]*)\]\[\s*' + re.escape(key) + r'\s*\]')
        texto_actualizado = ref_pattern.sub(r'![\1](' + ruta + r')', texto_actualizado)

    # Escribir el resultado en el mismo archivo (o uno nuevo, si prefieres)
    nuevo_path = path_md.with_name(path_md.stem + ".md")
    nuevo_path.write_text(texto_actualizado, encoding="utf-8")

    print(f"✨ Archivo procesado guardado como: {nuevo_path}")
    print(f"🖼️  Imágenes extraídas en: {carpeta_destino.resolve()}")

    return nuevo_path

def enrich_markdown(md_text: str, image_descriptions: dict) -> str:
    """
    Inserta la descripción debajo de cada referencia ![...](...) si existe en image_descriptions.
    """

    # Patrón para ![alt](ruta)
    #img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    img_pattern = re.compile(r'!\[(.*?)\]\((.*?\.\w{3,4})\)')


    def replace_img(match):
        alt_text, img_path = match.groups()
        # Normalizar ruta para buscar en el dict (decodificar %20)
        img_path_norm = img_path.replace("%20", " ")
        desc = image_descriptions.get(img_path_norm)
        if not desc:
            # fallback: tratar de normalizar separadores
            key_candidates = [img_path, img_path_norm]
            for k in key_candidates:
                if k in image_descriptions:
                    desc = image_descriptions[k]
                    break

        # Reconstruir línea de imagen + descripción si existe
        if desc:
            return f"![{img_path_norm}](...)\n\n> *Descripción de {img_path_norm}:* {desc}\n"
        else:
            return match.group(0)  # sin cambios

    return img_pattern.sub(replace_img, md_text)

def unzip_markdown(zip_path = "/content/NICOLAS ALEJANDRO FREZ VALENCIA_1967793_assignsubmission_file_Reporte1.zip"):
  # Crear carpeta temporal y descomprimir
  temp_dir = tempfile.mkdtemp()
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:
      zip_ref.extractall(temp_dir)
  return temp_dir


from pathlib import Path
from markdown import markdown
from markdown_it import MarkdownIt
from weasyprint import HTML
from IPython.display import Markdown, display

import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import contextlib
import os, sys

import os, sys, logging, warnings

class ultra_quiet:
    def __enter__(self):
        # Silenciar logging y warnings de Python
        logging.disable(logging.CRITICAL)
        for name in ("weasyprint", "fontTools", "fontTools.subset",
                     "fontTools.ttLib", "fontTools.ttLib.ttFont"):
            lg = logging.getLogger(name)
            lg.disabled = True
            lg.propagate = False
        warnings.filterwarnings("ignore")

        # Guardar stdout/stderr de alto nivel
        self._stdout = sys.stdout
        self._stderr = sys.stderr

        # Guardar FDs originales
        self.stdout_fd = os.dup(1)
        self.stderr_fd = os.dup(2)

        # Abrir /dev/null y redirigir FDs (cubre prints y C-extensions)
        self.null = open(os.devnull, 'w')
        os.dup2(self.null.fileno(), 1)
        os.dup2(self.null.fileno(), 2)

        # También redirigir objetos de alto nivel
        sys.stdout = self.null
        sys.stderr = self.null
        return self

    def __exit__(self, exc_type, exc, tb):
        # Restaurar Python-level
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        logging.disable(logging.NOTSET)

        # Restaurar FDs
        os.dup2(self.stdout_fd, 1)
        os.dup2(self.stderr_fd, 2)
        os.close(self.stdout_fd)
        os.close(self.stderr_fd)

        # Cerrar /dev/null
        self.null.close()


def markdown_to_pdf(answer, output_path, display_=False):
    try:
        if display_:
            display(Markdown(f"File: {output_path}\n{answer}\n-----"))

        # Convertir markdown a HTML con envoltorio bonito
        #html_body = markdown(answer, extensions=["extra", "tables", "sane_lists"])

        md = MarkdownIt()
        html_body = md.render(answer)

        html_full = f"""
<html><head><style>
body {{
    font-family: 'Helvetica', sans-serif;
    margin: 2em;
    line-height: 1.6;
    color: #222;
}}
h1, h2, h3 {{
    color: #004d99;
}}
pre, code {{
    background-color: #f5f5f5;
    padding: 0.2em 0.4em;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
}}
blockquote {{
    border-left: 4px solid #ccc;
    margin: 1em 0;
    padding-left: 1em;
    color: #555;
}}
ul, ol {{
    margin-left: 1.5em;
    padding-left: 1em;
}}
li {{
    margin-bottom: 0.2em;
}}
</style></head><body>
{html_body}
</body></html>
"""
        # Guardar como PDF
        pdf_path = f"{output_path}"


        with ultra_quiet():
          HTML(string=html_full).write_pdf(pdf_path)
        #HTML(string=html_full).write_pdf(pdf_path)
        #print(2)
 
        #HTML(string=html_full).write_pdf(pdf_path)

        print(f"[✔] PDF guardado en: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"[⚠] Error en archivo {pdf_path}: {e}")
        return None


def list_files_in_drive_folder(folder_id):
    """Devuelve lista de (nombre, id) de los archivos en una carpeta de Drive."""
    creds, _ = google.auth.default()
    drive = build("drive", "v3", credentials=creds)

    files_info = dict()
    page_token = None

    while True:
        response = drive.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            spaces="drive",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()

        for file in response.get("files", []):
            files_info[file["name"].split('.')[0]] = file["id"]

        page_token = response.get("nextPageToken", None)
        if page_token is None:
            break

    return files_info

