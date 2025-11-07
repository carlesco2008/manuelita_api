

# ==============================================================================
# MANUELITA CHATBOT — Módulo 2: Agente Conversacional (Versión Final v4 - Corregida)
## Objetivo:
# - Corregir el error de inicialización 'missing tool_names' de forma definitiva. (YA CORREGIDO)
# - CORREGIR EL ERROR DE INDENTACIÓN en buscar_datos_especificos. (APLICADO AQUÍ)
# - Mantener el enrutamiento robusto y las optimizaciones de velocidad.
# ==============================================================================
import gradio as gr
import os
import json
# --- LangChain / Infra RAG y Agente ---
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

print("Starting Manuelita Conversational Agent...")
agent_executor = None
initialization_error = None



load_dotenv()


# ==============================================================================
# HERRAMIENTA 2: BÚSQUEDA EN DATOS ESTRUCTURADOS (JSON)
# 
# !!! CORRECCIÓN CRÍTICA DE INDENTACIÓN APLICADA AQUÍ !!!
# ==============================================================================
def buscar_datos_especificos(pregunta: str) -> str:
    print(f"DEBUG: [Tool Called] Using structured data tool for: '{pregunta}'")
    pregunta = pregunta.lower()
    
    # 1. Carga de datos (Try/Except)
    try:
        with open('datos_estructurados.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return "Error: El archivo 'datos_estructurados.json' no fue encontrado."
    
    # 2. Lógica de Búsqueda (Se ejecuta SOLO si la carga es exitosa)
    if any(keyword in pregunta for keyword in ["teléfono", "contacto", "llamar", "correo", "email"]):
        # Retorna toda la sección de 'contacto' para que el LLM extraiga el teléfono
        return json.dumps(data.get("contacto", {"info": "No se encontró información de contacto."}))
        
    if any(keyword in pregunta for keyword in ["horario", "atención", "abren", "cierran"]):
        return json.dumps(data.get("horarios", {"info": "No se encontraron horarios de atención."}))
        
    if any(keyword in pregunta for keyword in ["sedes", "dirección", "ubicación", "oficina"]):
        return json.dumps(data.get("sedes_cali", [{"info": "No se encontraron sedes."}]))
        
    if "nit" in pregunta:
        # Aquí podemos devolver solo el NIT, ya que es un dato puntual
        return data.get("contacto", {}).get("nit", "NIT no encontrado.")
        
    # Fallback
    return "No encontré datos específicos para esa pregunta. La pregunta puede ser demasiado general para esta herramienta."


try:
    # --------------------------------------------------------------------------
    # 1) API Key (Gemini)
    # --------------------------------------------------------------------------
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("CRITICAL: Missing GOOGLE_API_KEY.")
        
    # --------------------------------------------------------------------------
    # 2) Carga y Preparación de la Base de Conocimiento (RAG)
    # --------------------------------------------------------------------------
    print("Loading Markdown documents from 'data/raw/'...")
    loader = DirectoryLoader(path="data/raw/", glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    if not docs:
        raise ValueError("No .md files found in 'data/raw/'.")
        
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    splits = markdown_splitter.split_text("\n".join([doc.page_content for doc in docs]))
    if not splits:
        raise ValueError("Markdown splitting yielded no chunks.")
    print(f"DEBUG: Loaded {len(docs)} docs -> {len(splits)} chunks.")
    
    # --------------------------------------------------------------------------
    # 3) Búsqueda Híbrida y Re-ranking
    # --------------------------------------------------------------------------
    print("Building hybrid retriever with re-ranker...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    keyword_retriever = BM25Retriever.from_documents(splits)
    keyword_retriever.k = 7
    ensemble_retriever = EnsembleRetriever(retrievers=[semantic_retriever, keyword_retriever], weights=[0.75, 0.25])
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
    reranking_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
    
    # --------------------------------------------------------------------------
    # 4) LLM (Optimizado para Velocidad y Precisión)
    # --------------------------------------------------------------------------
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0, google_api_key=api_key)
    
    # ==========================================================================
    # HERRAMIENTA 1: SISTEMA RAG
    # ==========================================================================
    rag_chain_prompt = PromptTemplate.from_template("Contexto:{context}\n\nPregunta:{input}\n\nRespuesta concisa:")
    rag_chain_internal = create_stuff_documents_chain(llm, rag_chain_prompt)
    retrieval_chain_for_tool = create_retrieval_chain(reranking_retriever, rag_chain_internal)
    
    def invocar_cadena_rag(pregunta: str) -> str:
        print(f"DEBUG: [Tool Called] Using RAG tool for: '{pregunta}'")
        response = retrieval_chain_for_tool.invoke({"input": pregunta})
        return response["answer"]
        
    # --------------------------------------------------------------------------
    # 5) Lista de Herramientas (Descripciones Claras)
    # --------------------------------------------------------------------------
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
    
    # --------------------------------------------------------------------------
    # 6) Prompt para el Agente (¡ESTRUCTURA CORREGIDA!)
    # --------------------------------------------------------------------------
    agent_prompt_template = """Eres un asistente experto en enrutar preguntas. Tu única misión es elegir la herramienta correcta. Sigue estas reglas estrictamente.

**REGLAS DE DECISIÓN:**
1.  **Lee la PREGUNTA ACTUAL del usuario.**
2.  **Revisa la descripción de las herramientas.** Si la pregunta contiene palabras como 'teléfono', 'contacto', 'NIT', 'horario', 'dirección', DEBES usar la herramienta `buscar_datos_especificos_manuelita`.
3.  **Si la regla 2 no aplica,** entonces la pregunta es general. DEBES usar la herramienta `busqueda_documental_manuelita`.
4.  Una vez que la herramienta te dé una `Observation`, tu trabajo es simple: preséntala al usuario de forma clara en la `Final Answer`. No vuelvas a pensar ni a usar otra herramienta.

**HERRAMIENTAS DISPONIBLES:**
Debes elegir una de las siguientes herramientas: {tool_names}
Aquí están sus descripciones:{tools}

**FORMATO OBLIGATORIO:**
Thought: Mi análisis de la pregunta y la herramienta que debo usar según las reglas.
Action: El nombre de la herramienta elegida.
Action Input: La pregunta del usuario.
Observation: El resultado que la herramienta me da.
Thought: Tengo el resultado. Ahora lo presentaré al usuario.
Final Answer: La respuesta final en español.

--- COMIENZA ---
**HISTORIAL (para contexto):** {chat_history}
**PREGUNTA ACTUAL:** {input}
**TU RAZONAMIENTO:** {agent_scratchpad}"""

    # Se crea el prompt a partir del template.
    agent_prompt = PromptTemplate.from_template(agent_prompt_template)
    
    # ¡¡LA CORRECCIÓN CRÍTICA PARA EL ERROR 'missing tool_names'!!
    # Usamos .partial() para pre-cargar el prompt con la información de las herramientas.
    agent_prompt = agent_prompt.partial(
        tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools]),
    )
    
    # --------------------------------------------------------------------------
    # 7) Implementación de Memoria y Creación del Agente
    # --------------------------------------------------------------------------
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    agent = create_react_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6 
    )
    print("Manuelita Conversational Agent is ready ✅")
    
except Exception as e:
    initialization_error = e
    print(f"CRITICAL init error: {e}")

# ------------------------------------------------------------------------------
# 8) Función de respuesta para Gradio
# ------------------------------------------------------------------------------
def get_agent_response(message, history):
    if initialization_error:
        return f"Error de inicialización: {initialization_error}"
    if not agent_executor:
        return "Error: El agente conversacional no está disponible."
        
    try:
        # Se convierte el historial de Gradio al formato esperado por LangChain/Memory
        # Gradio lo maneja como una lista de tuplas (user_msg, bot_msg)
        # LangChain Memory lo maneja internamente. Solo pasamos el nuevo input.
        
        response = agent_executor.invoke({"input": message})
        return response["output"]
    except Exception as e:
        print(f"[ERROR] Procesando con el agente: {e}")
        return f"Lo siento, ocurrió un error. Detalle: {e}"

# ------------------------------------------------------------------------------
# 9) Interfaz Gradio
# ------------------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=get_agent_response,
    title="Manuelita — Agente Conversacional (Módulo 2)",
    description="Haz tu consulta sobre Manuelita. El asistente decidirá si usar la búsqueda documental (RAG) o consultar datos estructurados para responder.",
    examples=[
        ["¿Qué productos de energías renovables ofrece Manuelita?"],
        ["¿Cuál es el NIT de la empresa?"],
        ["Háblame sobre las uvas que producen"],
        ["Y cuál es el número de teléfono de servicio al cliente?"],
    ],
    theme="soft",
    chatbot=gr.Chatbot(height=550),
)

# ------------------------------------------------------------------------------
# 10) Lanzamiento
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Asegúrate de tener la carpeta 'data/raw/' con archivos .md
    # y el archivo 'datos_estructurados.json' en el mismo directorio que este script.
    demo.launch()