import os
import requests

from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

from langchain.text_splitter import TokenTextSplitter

from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.callbacks.stdout import StdOutCallbackHandler
# Callbacks para streaming
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Document loaders & splitters
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & vectorstore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Chat model, prompts y cadenas
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
import math


def eval_fragment(guideline, report_extract, model_name="gpt-4.1-mini", temperature=0.1):

    # 8️⃣ Prompt template
    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(
            #"# Reporte del Profesor (ejemplo)\n{example}# Reporte Estudiante:\n{report_extract}\n\n{guideline}"
            "# Reporte Estudiante:\n{report_extract}\n\n{guideline}"

        )
    ])

    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    # Formatear el prompt con las variables proporcionadas
    formatted_prompt = prompt.format(
        report_extract=report_extract,
        #filtered_text=full_text,
        guideline=guideline
    )

    # Imprimir el prompt formateado
    #print("Prompt final enviado al modelo:")
    #print(formatted_prompt)

    chain = prompt | llm

    try:
        respuesta = chain.invoke(
            {"report_extract": report_extract, "guideline": guideline}
        )

        respuesta = respuesta.content
    except Exception as e:
        respuesta = f"❌ Error la guía: {guideline}\n{e}"

    return respuesta

from openai import OpenAI
import re
import json
import os

def extraer_encabezados_markdown(md_text, preview_len=150):
    """
    Extrae encabezados de un texto markdown junto con su número de línea.
    Devuelve una lista de tuplas: (preview del encabezado y contenido, número de línea)
    Detecta encabezados que comiencen con '#' o con '**...'.
    """
    resultado = []
    lineas = md_text.splitlines()
    char_pos = 0  # posición acumulada en caracteres

    for i, linea in enumerate(lineas):
        linea_stripped = linea.lstrip()
        if linea_stripped.startswith("#") or linea_stripped.startswith("**..."):
            # Extraer los siguientes caracteres desde la posición actual
            inicio = char_pos
            preview = md_text[inicio:inicio + preview_len] + "..."
            resultado.append((preview, i))
        # Avanzar la posición acumulada (+1 por el salto de línea que splitlines() elimina)
        char_pos += len(linea) + 1

    resultado.append(("EOF", len(lineas)))
    return resultado


def get_sections(client, encabezados, sections=["Introducción","Conclusión"], verbose=False):
  segmentos = "\n\n(...)\n\n".join([f"[Linea {i}]\n{doc}" for doc, i in encabezados])


  prompt_header = (
        'A continuación se presentan segmentos de un informe académico (en número de lineas).\n'+
        '### Informe:\n'+
        '"""\n{segmentos}\n\n(...)"""'
  )

  prompt_header = prompt_header.format(segmentos=segmentos)

  prompt_header += (
        "### Instrucciones \n\nTu tarea es identificar entre qué lineas " +
        "se podría encontrar cada una de los siguientes contenidos:\n\n" +
        "\n".join([f"- {s[0]} ({s[1]})" for s in sections]) +
        "\n\nPara cada sección, incluye todas las líneas desde el inicio "+
        "de la sección **hasta la línea del encabezado correspondiente al siguiente contenido**, "+
        "sin omitir líneas intermedias, aunque parezcan vacías o de transición.\n"+
        "Ten en cuenta que secciones de ejemplo deben estar definitivamente relacionada al contenido previo.\n"+
        "Si un contenido definitivamente crees que no se encuentra indícalo con null.\n"
  )

  cierre = "\n\n### Formato de respuesta (json):\n\n```json\n{\n   " + "\n   ".join(
        [f"\"{s[1]}\": [inicio,fin], #or null" for s in sections[:4]]
  )+"\n   ...\n}\n```"

  full_prompt = f"{prompt_header}{cierre}"

  if verbose: print(full_prompt)


  response = client.chat.completions.create(
      model="gpt-4.1",
      messages=[{"role": "user", "content": full_prompt}],
      temperature=0.1,
      timeout=60  # segundos
)
  # Parsear la respuesta:
  # extraer bloque ```
  answer_gpt = response.choices[0].message.content.strip()

  #print(answer_gpt)

  # Buscar bloque indentado con llaves
  match = re.search(r"```json(.*?)```", answer_gpt, re.DOTALL)
  if match:
      code = match.group(1)
      indice = json.loads(code)


  for key, value in indice.items():
      if value is not None:
        indice[key] = [i+1 if i is not None else None for i in value]

  indice["eof"] = encabezados[-1][1]+1

  return indice

#open file DiseñoAlgoritmos
import ast

def generate_guideline(filename, guideline_template):
  ruta = f'/content/evaluacionInformes/ListasCotejo/{filename}'

  with open(ruta, 'r') as file:
      # Leer todas las líneas
      lines = file.readlines()

  # 1. Obtener claves desde la primera línea
  sections_to_eval = ast.literal_eval(lines[0].strip())  # Convierte de string a lista

  # 2. Unir el resto como texto del cotejo
  cotejo = "".join(lines[1:]).strip()

  #with open(f'/content/evaluacionInformes/Ejemplos/{filename}', 'r') as file:
  #    ejemplo = file.read()
  ejemplo = None

  return guideline_template.format(ListaCotejo=cotejo), ejemplo, sections_to_eval

def merge_intervals(intervals):
    # 1. Ordenar por inicio
    intervals.sort(key=lambda x: x[0])

    merged = []

    for current in intervals:
        if not merged:
            merged.append(current)
        else:
            last = merged[-1]
            # 2. Verificar traslape o inclusión
            if current[0] <= last[1]:  # traslape o incluido
                # 3. Unir intervalos
                last[1] = max(last[1], current[1])
            else:
                merged.append(current)

    return merged

def generate_text(contenido, sections, indice):
  lines = contenido.splitlines()

  text = "Estos son algunas secciones seleccionadas del informe entregado por el estudiante:\n\n"

  selected_lines = []
  for section in sections:
      intervalos = indice[section]
      if isinstance(intervalos[0], int):  # Un solo intervalo tipo [start, end]
          selected_lines.append([intervalos[0] - 1, intervalos[1]])
      else:  # Lista de intervalos
          for start, end in intervalos:
              selected_lines.append([start - 1, end])

  selected_lines = merge_intervals(selected_lines)

  selected_texts = []
  for line_range in selected_lines:
      start_index, end_index = line_range
      selected_text = "\n".join(lines[max(start_index, 0) : min(end_index, len(lines))])
      selected_texts.append(selected_text)

  for i, tt in enumerate(selected_texts):
      text += f"(...)\n{tt}\n\n"

  return text
