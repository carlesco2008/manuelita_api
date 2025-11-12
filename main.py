# ==============================================================================
# MANUELITA CHATBOT — API con FastAPI y Qdrant
# Objetivo:
# - Servir el agente conversacional a través de un endpoint de API.
# - Consultar una base de datos vectorial Qdrant preexistente.
# - No construir la base de datos en tiempo de ejecución.
# - Manejar historiales de conversación por sesión.
# ==============================================================================
import os
import json
from dotenv import load_dotenv

# --- Framework API ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- LangChain / Infra RAG y Agente ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

print("Iniciando la configuración del Agente Conversacional para API...")

# --- Carga de variables de entorno ---
load_dotenv()

# ==============================================================================
# 1. Configuración y Carga de Credenciales
# ==============================================================================
try:
    # --- Clave de API de Gemini ---
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("CRÍTICO: Falta la variable de entorno GOOGLE_API_KEY.")

    # --- Configuración de Qdrant ---
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    if not qdrant_url or not qdrant_collection_name:
        raise ValueError("CRÍTICO: Faltan las variables de entorno QDRANT_URL o QDRANT_COLLECTION_NAME.")
    
    print("Credenciales y configuración cargadas exitosamente.")

except ValueError as e:
    print(e)
    # Si hay un error crítico en la configuración, detenemos la aplicación.
    exit()


# ==============================================================================
# 2. Definición de Herramientas
# ==============================================================================

# HERRAMIENTA 1: BÚSQUEDA EN DATOS ESTRUCTURADOS (JSON)
def buscar_datos_especificos(pregunta: str) -> str:
    print(f"DEBUG: [Tool Called] Usando herramienta de datos estructurados para: '{pregunta}'")
    pregunta = pregunta.lower()
    try:
        with open('datos_estructurados.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return "Error: El archivo 'datos_estructurados.json' no fue encontrado."
    
    if any(keyword in pregunta for keyword in ["teléfono", "contacto", "llamar", "correo", "email"]):
        return json.dumps(data.get("contacto", {"info": "No se encontró información de contacto."}))
    if any(keyword in pregunta for keyword in ["horario", "atención", "abren", "cierran"]):
        return json.dumps(data.get("horarios", {"info": "No se encontraron horarios de atención."}))
    if any(keyword in pregunta for keyword in ["sedes", "dirección", "ubicación", "oficina"]):
        return json.dumps(data.get("sedes_cali", [{"info": "No se encontraron sedes."}]))
    if "nit" in pregunta:
        return data.get("contacto", {}).get("nit", "NIT no encontrado.")
        
    return "No encontré datos específicos para esa pregunta. La pregunta puede ser demasiado general para esta herramienta."


# ==============================================================================
# 3. Inicialización del Agente (se ejecuta una sola vez al iniciar la API)
# ==============================================================================

# Almacenamiento en memoria para las conversaciones de diferentes sesiones
conversations = {}

# --- Modelos y Componentes ---
print("Inicializando modelos y componentes de LangChain...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0, google_api_key=gemini_api_key)

# --- Conexión a Qdrant ---
print(f"Conectando a Qdrant en {qdrant_url} y a la colección '{qdrant_collection_name}'...")
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Se crea el objeto vectorstore que SOLO CONSULTA la colección existente
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=qdrant_collection_name,
    embeddings=embedding_model,
)

# --- Creación del Retriever con Re-ranking ---
qdrant_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
reranking_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=qdrant_retriever)
print("Retriever de Qdrant con re-ranking configurado.")

# HERRAMIENTA 2: SISTEMA RAG (que ahora usa Qdrant)
rag_chain_prompt = PromptTemplate.from_template("Contexto:{context}\n\nPregunta:{input}\n\nRespuesta concisa:")
rag_chain_internal = create_stuff_documents_chain(llm, rag_chain_prompt)
retrieval_chain_for_tool = create_retrieval_chain(reranking_retriever, rag_chain_internal)

def invocar_cadena_rag(pregunta: str) -> str:
    print(f"DEBUG: [Tool Called] Usando herramienta RAG (Qdrant) para: '{pregunta}'")
    response = retrieval_chain_for_tool.invoke({"input": pregunta})
    return response["answer"]

# --- Lista de Herramientas para el Agente ---
tools = [
    Tool(
        name="buscar_datos_especificos_manuelita",
        func=buscar_datos_especificos,
        description=(
            "OBLIGATORIA si la pregunta del usuario contiene palabras clave como: 'teléfono', 'contacto', 'NIT', 'horario', 'dirección', 'ubicación', 'correo', 'llamar', 'sede', 'oficina'. "
            "Es para obtener datos puntuales y estructurados. NO la uses para preguntas generales."
        )
    ),
    Tool(
        name="busqueda_documental_manuelita",
        func=invocar_cadena_rag,
        description=(
            "Úsala para TODAS las demás preguntas que SEAN generales sobre la empresa Manuelita. "
            "Por ejemplo: '¿cuál es la historia de la empresa?', 'háblame de sus productos', '¿qué hacen en sostenibilidad?'. "
            "Esta es tu herramienta por defecto a menos que la pregunta sea sobre datos de contacto específicos."
        )
    ),
]

# --- Prompt del Agente (sin cambios) ---
agent_prompt_template = """
# IDENTIDAD Y MISIÓN
Eres ManuelitaGPT, un asistente experto y ultra-preciso de la empresa Manuelita. Tu única misión es analizar la pregunta del usuario y usar la herramienta correcta para encontrar la respuesta. Eres metódico y sigues las reglas al pie de la letra.

# HERRAMIENTAS DISPONIBLES
Tienes acceso EXCLUSIVO a las siguientes herramientas. No intentes usar ninguna otra.

{tools}

# FORMATO ESTRICTO Y OBLIGATORIO
Debes usar el siguiente formato para tu razonamiento. No te desvíes NUNCA de este formato.

Thought: (Aquí escribes tu razonamiento sobre qué paso tomar a continuación. Analiza la pregunta y decide qué herramienta es la más apropiada según su descripción).
Action: (El nombre de la única herramienta que has decidido usar. Debe ser una de [{tool_names}]).
Action Input: (La pregunta exacta o el término de búsqueda que le pasarás a la herramienta).
Observation: (El resultado que te devuelve la herramienta. No lo inventes, es lo que recibes).
... (este patrón de Thought/Action/Action Input/Observation puede repetirse si es necesario).
Thought: (Una vez que la `Observation` contiene la información suficiente para responder, escribes este pensamiento final).
Final Answer: (La respuesta final al usuario. Esta es la única parte que el usuario verá).

# REGLAS FINALES
1.  **Formato Inflexible:** Tu respuesta debe contener SÓLO el texto que sigue el formato anterior. NO añadas texto conversacional ni explicaciones fuera de la estructura `Thought`/`Action`/`Final Answer`.
2.  **Idioma:** La `Final Answer` debe ser siempre en **español**, de forma amable y profesional.
3.  **No Supongas:** Si la información no está en la `Observation`, indica que no pudiste encontrar la información. No inventes respuestas.

--- COMIENZA LA EJECUCIÓN ---

Historial de la Conversación (para darte contexto):
{chat_history}

Pregunta del Usuario: {input}

Tu Razonamiento (sigue el formato estricto):
{agent_scratchpad}
"""

agent_prompt = PromptTemplate.from_template(agent_prompt_template)
agent_prompt = agent_prompt.partial(
    tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
    tool_names=", ".join([tool.name for tool in tools]),
)

# --- Creación del Agente y Executor ---
# IMPORTANTE: No se añade memoria aquí para que el executor sea reutilizable y sin estado.
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # Déjalo en True para depurar, cámbialo a False en producción
    handle_parsing_errors=True,
    max_iterations=6
)

print("Agente Conversacional listo y configurado ✅")

# ==============================================================================
# 4. Definición de la API con FastAPI
# ==============================================================================
app = FastAPI(
    title="Manuelita Conversational Agent API",
    description="API para interactuar con el agente de Manuelita que utiliza RAG con Qdrant.",
    version="1.0.0",
)

class ChatRequest(BaseModel):
    message: str
    session_id: str # Identificador único para cada conversación de usuario

@app.post("/chat")
def chat(request: ChatRequest):
    """
    Recibe una pregunta y un ID de sesión, y devuelve la respuesta del agente.
    Maneja el historial de la conversación internamente.
    """
    print(f"Recibida petición para session_id: {request.session_id}")
    
    # Obtener o crear la memoria para la sesión del usuario
    if request.session_id not in conversations:
        print(f"Creando nueva memoria para session_id: {request.session_id}")
        conversations[request.session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    memory = conversations[request.session_id]
    
    try:
        # Invocamos al agente, pasándole la memoria específica de esta sesión
        response = agent_executor.invoke({
            "input": request.message,
            "chat_history": memory.chat_memory.messages
        })
        
        # Guardamos manualmente el contexto en la memoria de la sesión
        memory.save_context({"input": request.message}, {"output": response["output"]})
        
        return {"response": response["output"]}

    except Exception as e:
        print(f"[ERROR] Procesando con el agente para session_id {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al procesar la pregunta. Detalle: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "El agente conversacional de Manuelita está en línea."}

# Para ejecutar la aplicación localmente:
# uvicorn main:app --reload