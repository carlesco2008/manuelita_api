################################################################################
#                                                                              #
#                   MANUELITA CHATBOT - MÓDULO 2: AGENTE EXPERTO                 #
#                                                                              #
#   Versión: 5.0 "Espectacular"                                                #
#   Autor: [Tu Nombre/Equipo]                                                  #
#   Fecha: [Fecha Actual]                                                      #
#                                                                              #
#   Propósito: Un agente conversacional robusto que enruta inteligentemente    #
#              las consultas de los usuarios a la herramienta adecuada:        #
#              1. Búsqueda en base documental (RAG) para preguntas generales.  #
#              2. Consulta de datos estructurados para información específica. #
#                                                                              #
################################################################################

# ==============================================================================
# 0. LIBRERÍAS E IMPORTACIONES
# ==============================================================================
import gradio as gr
import os
import json
from dotenv import load_dotenv

# --- LangChain: El Ecosistema para construir con LLMs ---
# Componentes del RAG (Retrieval-Augmented Generation)
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma

# Modelos y Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI

# Componentes del Agente (El "cerebro" que elige herramientas)
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

print("✅ Librerías importadas correctamente.")

# ==============================================================================
# 1. CONFIGURACIÓN CENTRALIZADA
# ==============================================================================
class Config:
    """
    Clase para centralizar todas las configuraciones del agente.
    Modificar estos valores es la forma más fácil de experimentar.
    """
    # --- Modelos ---
    MODEL_LLM = "gemini-2.5-pro"
    MODEL_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_RERANKER = "BAAI/bge-reranker-base"
    
    # --- Rutas de Archivos ---
    PATH_KNOWLEDGE_BASE = "data/raw/"
    PATH_STRUCTURED_DATA = "datos_estructurados.json"
    
    # --- Parámetros del Retriever ---
    RETRIEVER_SEMANTIC_K = 7  # Documentos a obtener de la búsqueda semántica
    RETRIEVER_KEYWORD_K = 7   # Documentos a obtener de la búsqueda por palabras clave
    ENSEMBLE_WEIGHTS = [0.75, 0.25] # Ponderación: 75% semántico, 25% palabra clave
    RERANKER_TOP_N = 3        # Documentos finales a enviar al LLM después de re-rankear

    # --- Parámetros del Agente ---
    AGENT_MAX_ITERATIONS = 5
    AGENT_TEMPERATURE = 0.0   # 0.0 para máxima precisión y consistencia

print("✅ Configuración cargada.")

# ==============================================================================
# 2. DEFINICIÓN DE HERRAMIENTAS
# Un agente necesita herramientas para interactuar con el mundo.
# ==============================================================================

# ------------------------------------------------------------------------------
# HERRAMIENTA 1: BÚSQUEDA EN DATOS ESTRUCTURADOS (JSON)
# ------------------------------------------------------------------------------
def buscar_datos_especificos(pregunta: str) -> str:
    """
    Busca información puntual en un archivo JSON local.
    Diseñada para responder preguntas sobre datos de contacto, horarios y sedes.
    
    Args:
        pregunta (str): La pregunta del usuario.

    Returns:
        str: Una cadena en formato JSON con la información encontrada o un
             mensaje de error.
    """
    print(f"DEBUG: [Herramienta 'Datos Específicos'] -> Recibió la pregunta: '{pregunta}'")
    pregunta_lower = pregunta.lower()
    
    # --- Carga segura de los datos ---
    try:
        with open(Config.PATH_STRUCTURED_DATA, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return json.dumps({"error": f"Archivo no encontrado en '{Config.PATH_STRUCTURED_DATA}'."})
    except json.JSONDecodeError:
        return json.dumps({"error": "El archivo de datos estructurados no es un JSON válido."})

    # --- Lógica de enrutamiento por palabras clave ---
    if any(kw in pregunta_lower for kw in ["teléfono", "contacto", "llamar", "correo", "email"]):
        return json.dumps(data.get("contacto", {"info": "No encontré detalles de contacto."}))
        
    if any(kw in pregunta_lower for kw in ["horario", "atención", "abren", "cierran"]):
        return json.dumps(data.get("horarios", {"info": "No encontré información de horarios."}))
        
    if any(kw in pregunta_lower for kw in ["sedes", "dirección", "ubicación", "oficina"]):
        return json.dumps(data.get("sedes_cali", [{"info": "No encontré información sobre sedes."}]))
        
    if "nit" in pregunta_lower:
        return data.get("contacto", {}).get("nit", "No encontré el NIT en los datos.")
        
    return "Esta pregunta no parece ser sobre datos específicos (contacto, horario, NIT). Intenta con la otra herramienta."

# ------------------------------------------------------------------------------
# HERRAMIENTA 2: BÚSQUEDA DOCUMENTAL (SISTEMA RAG)
# ------------------------------------------------------------------------------
def crear_invocador_rag(retriever, llm):
    """
    Crea y devuelve una función que actúa como la herramienta RAG.
    Esto encapsula la cadena de recuperación para que sea fácil de llamar.
    """
    rag_prompt = PromptTemplate.from_template(
        "Responde la pregunta del usuario de forma concisa y clara, basándote únicamente en el siguiente contexto:\n\n"
        "--- CONTEXTO ---\n{context}\n--- FIN CONTEXTO ---\n\n"
        "Pregunta: {input}"
    )
    
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    def invocar_cadena_rag(pregunta: str) -> str:
        """
        Invoca la cadena RAG completa para responder preguntas generales.
        """
        print(f"DEBUG: [Herramienta 'RAG'] -> Recibió la pregunta: '{pregunta}'")
        response = retrieval_chain.invoke({"input": pregunta})
        return response.get("answer", "No pude generar una respuesta a partir de los documentos.")
        
    return invocar_cadena_rag

print("✅ Herramientas definidas.")

# ==============================================================================
# 3. LÓGICA DE INICIALIZACIÓN DEL AGENTE
# Aquí se construye el "cerebro" del chatbot, paso a paso.
# ==============================================================================
def inicializar_agente_completo():
    """
    Orquesta toda la construcción del agente: carga de datos, creación del
    retriever, instanciación del LLM y ensamblaje final del AgentExecutor.
    """
    try:
        # --- PASO 1: Cargar la API Key ---
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Error crítico: La variable de entorno GOOGLE_API_KEY no está configurada.")
        
        # --- PASO 2: Cargar y procesar la base de conocimiento para RAG ---
        print("INFO: Cargando documentos Markdown...")
        loader = DirectoryLoader(path=Config.PATH_KNOWLEDGE_BASE, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        docs = loader.load()
        if not docs:
            raise FileNotFoundError(f"No se encontraron archivos .md en '{Config.PATH_KNOWLEDGE_BASE}'.")
        
        headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        splits = markdown_splitter.split_text("\n".join([doc.page_content for doc in docs]))
        print(f"INFO: {len(docs)} documentos -> {len(splits)} fragmentos (chunks).")
        
        # --- PASO 3: Construir el Retriever Híbrido con Re-ranking ---
        # Este es el corazón del sistema RAG. Combina lo mejor de dos mundos:
        # - Búsqueda Semántica (por significado) con ChromaDB y embeddings.
        # - Búsqueda por Palabras Clave (lexical) con BM25.
        # - Re-ranking para mejorar la precisión de los resultados finales.
        print("INFO: Construyendo el retriever híbrido...")
        embedding_model = HuggingFaceEmbeddings(model_name=Config.MODEL_EMBEDDING)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
        
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": Config.RETRIEVER_SEMANTIC_K})
        keyword_retriever = BM25Retriever.from_documents(splits)
        keyword_retriever.k = Config.RETRIEVER_KEYWORD_K
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, keyword_retriever], 
            weights=Config.ENSEMBLE_WEIGHTS
        )
        
        reranker = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name=Config.MODEL_RERANKER), top_n=Config.RERANKER_TOP_N)
        reranking_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=ensemble_retriever)
        
        # --- PASO 4: Instanciar el LLM ---
        llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_LLM,
            temperature=Config.AGENT_TEMPERATURE,
            google_api_key=api_key
        )
        
        # --- PASO 5: Ensamblar las herramientas para el Agente ---
        invocar_rag = crear_invocador_rag(reranking_retriever, llm)
        tools = [
            Tool(
                name="Consultar_Datos_Puntuales_Manuelita",
                func=buscar_datos_especificos,
                description=(
                    "INDISPENSABLE para obtener datos de contacto (teléfono, correo, NIT), horarios de atención o direcciones/sedes. "
                    "Úsala si la pregunta contiene palabras como 'teléfono', 'contacto', 'NIT', 'horario', 'dirección', 'ubicación', 'sede', 'oficina'. "
                    "NO la uses para preguntas abiertas o generales."
                )
            ),
            Tool(
                name="Buscar_Informacion_General_Manuelita",
                func=invocar_rag,
                description=(
                    "HERRAMIENTA PRINCIPAL para todas las preguntas generales sobre la empresa Manuelita. "
                    "Úsala para temas como: historia de la empresa, productos, sostenibilidad, informes, etc. "
                    "Esta es tu opción por defecto si la pregunta no es sobre datos de contacto específicos."
                )
            ),
        ]
        
        # --- PASO 6: Crear el Prompt del Agente (las instrucciones del cerebro) ---
        agent_prompt_template = """
        Eres 'Manuelita Asistente', un chatbot experto en la empresa Manuelita. Tu única tarea es analizar la pregunta del usuario y elegir la herramienta correcta para responderla. Sigue estas reglas al pie de la letra.

        **REGLAS DE DECISIÓN INFALIBLES:**
        1.  **Analiza la PREGUNTA ACTUAL:** Lee la pregunta del usuario cuidadosamente.
        2.  **Verifica si es sobre datos puntuales:** Si la pregunta contiene palabras clave como 'teléfono', 'contacto', 'NIT', 'horario' o 'dirección', DEBES usar la herramienta `Consultar_Datos_Puntuales_Manuelita`. SIN EXCEPCIONES.
        3.  **Para todo lo demás, usa la búsqueda general:** Si la regla 2 no se cumple, la pregunta es de conocimiento general. DEBES usar la herramienta `Buscar_Informacion_General_Manuelita`.
        4.  **Entrega la respuesta:** Una vez que la herramienta te dé un resultado ('Observation'), tu único trabajo es presentar esa información al usuario de manera amable en la 'Final Answer'. NO uses otra herramienta después de obtener un resultado.

        **HERRAMIENTAS DISPONIBLES:**
        {tools}

        **FORMATO DE RESPUESTA OBLIGATORIO:**
        Thought: [Tu razonamiento paso a paso sobre qué herramienta elegir según las reglas].
        Action: [El nombre EXACTO de la herramienta que elegiste].
        Action Input: [La pregunta original del usuario].
        Observation: [El resultado que te devuelve la herramienta].
        Thought: Tengo el resultado final. Ahora lo presentaré al usuario.
        Final Answer: [La respuesta final para el usuario, en español, clara y amable].

        --- ¡EMPIEZA AHORA! ---
        **Historial de la conversación:**
        {chat_history}

        **Pregunta Actual:** {input}

        **Tu Proceso de Razonamiento:**
        {agent_scratchpad}
        """
        
        agent_prompt = PromptTemplate.from_template(agent_prompt_template).partial(
            tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            tool_names=", ".join([tool.name for tool in tools]),
        )
        
        # --- PASO 7: Crear el Agente y el Executor ---
        agent = create_react_agent(llm, tools, agent_prompt)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors="Por favor, reformula tu pregunta. Tuve un problema para procesarla.",
            max_iterations=Config.AGENT_MAX_ITERATIONS
        )
        
        print("✅ Agente 'Manuelita Asistente' inicializado y listo para operar.")
        return agent_executor, None

    except Exception as e:
        print(f"CRITICAL: Error durante la inicialización -> {e}")
        return None, e

# ==============================================================================
# 4. LÓGICA DE INTERFAZ Y EJECUCIÓN
# ==============================================================================
agent_executor, initialization_error = inicializar_agente_completo()

def get_agent_response(message, history):
    """
    Función de callback para Gradio. Gestiona la interacción con el agente.
    """
    if initialization_error:
        return f"🔴 **Error Crítico de Inicialización** 🔴\n\nNo pude iniciar. Razón: {initialization_error}"
    if not agent_executor:
        return "🔴 **Error** 🔴\n\nEl agente no está disponible. Revisa los logs del servidor."
        
    try:
        response = agent_executor.invoke({"input": message})
        return response["output"]
    except Exception as e:
        print(f"[ERROR EN EJECUCIÓN] -> {e}")
        return f"Lo siento, encontré un problema inesperado al procesar tu solicitud. Detalles: {e}"

def main():
    """
    Función principal que lanza la interfaz de Gradio.
    """
    print("🚀 Lanzando la interfaz de Gradio...")
    
    # --- Creación de la Interfaz con Gradio ---
    demo = gr.ChatInterface(
        fn=get_agent_response,
        title="🤖 Manuelita Asistente Experto 🤖",
        description=(
            "¡Hola! Soy un asistente virtual especializado en Manuelita. "
            "Puedo buscar en nuestra base de conocimiento o consultar datos específicos como teléfonos y direcciones. ¿En qué te puedo ayudar?"
        ),
        examples=[
            ["¿Qué tipos de uva de mesa cultivan?"],
            ["¿Cuál es el NIT de la empresa y el teléfono de contacto?"],
            ["Háblame sobre la historia y fundación de Manuelita"],
            ["¿Tienen oficinas en Cali? ¿Cuál es la dirección?"],
        ],
        theme="soft",
        chatbot=gr.Chatbot(height=600, label="Conversación con Manuelita"),
        textbox=gr.Textbox(placeholder="Escribe tu pregunta aquí...", label="Tu Consulta"),
        submit_btn="Enviar Consulta",
        clear_btn="Limpiar Conversación",
    )
    
    # --- Lanzamiento de la App ---
    # `share=True` crea un enlace público temporal si lo necesitas.
    demo.launch()

# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================
if __name__ == "__main__":
    # Este bloque asegura que el código solo se ejecute cuando el script
    # es llamado directamente.
    main()