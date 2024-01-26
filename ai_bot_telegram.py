import openai
import requests
import time
####
import os
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from rich.console import Console
from utils import (DocsJSONLLoader, get_file_path, get_openai_api_key)

OPENAI_API_KEY = "YOUR KEY OPENAI"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
TOKEN_TELEGRAM = "YOUR KEY TELEGRAM"

console = Console()

recreate_chroma_db = False
fpath=get_file_path()

def get_updates(offset):
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/getUpdates"
    params = {"timeout": 100, "offset": offset}
    response = requests.get(url, params=params)
    return response.json()["result"]

####
def load_documents(file_path):
  loader = DocsJSONLLoader(file_path)
  data = loader.load()

  if not data:
        console.print("[red]Error: El archivo JSONL no contiene datos válidos.[/red]")
        return []
  
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1600,
    length_function=len,
    chunk_overlap=160 
  )
  return text_splitter.split_documents(data) if data else []

# CHROMA DB
def get_chroma_db(embeddings, documents, path):
    if recreate_chroma_db:
        console.print("Recreando Chroma DB")
        return Chroma.from_documents(
            documents=documents, embedding=embeddings, persist_directory=path
        )
    else:
        console.print("Cargando Chroma existente")
        return Chroma(persist_directory=path, embedding_function=embeddings)

def process_qa_query(query, retriever, llm):

    if query == "/ayuda":
        response = "Este bot esta orientado a brindar ayuda sobre coneptos de Inteligencia artificial. Solo escriba su pregunta y vere como ayudarlo"
        return response
    if query == "/start":
        response = "Bievenido al chat bot en conceptos de Inteligencia Artificial. Escriba su pregunta y vere como ayudarlo. Si necesita ayuda sobre el uso escriba /ayuda"
        return response
    else :
        ## Para emular consulta a la IA y hacer pruebas quitar las 2 lineas comentadas
        ##response ="RESPUESTA IA"
        ##return response
        console.print("[yellow]La IA está pensando...[/yellow]")
        console.print("Consulta recibida:", query)
        aclaracion = ". Es Obligatoria validar si la pregunta no esta relacionado en la tematica de Inteligencia artificial, responder: Su consulta no puede ser procesada."
        response = retriever.run(query+aclaracion)
        console.print("Respuesta obtenida de Chroma:", response)
        
        if response:
            return response
        else :
            return "sin respuesta"


####################
#MARCANDO MENSAJES PENDIENTES COMO LEIDOS PARA INICIAR CON EL BOT
def mark_new_messages():
    offset = None
    updates = get_updates(offset)
    for update in updates:
        print('-----')
        print(update["update_id"])
        offset = update["update_id"] + 1
    updates = get_updates(offset)  


def send_messages(chat_id, text):
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
    params = {"chat_id": chat_id, "text": text}
    response = requests.post(url, params=params)
    return response

def main():
    print("Starting bot...")
    #######
    print("Cantidad de documents")
    documents = load_documents(fpath)
    print(len(documents))
    get_openai_api_key()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # print(embeddings)
    vectorstore_chroma = get_chroma_db(embeddings, documents, "chroma_docs")
    vectorstore_chroma.embedding_function = embeddings  # Configurar embedding_function con las embeddings originales
    retriever = vectorstore_chroma.as_retriever(search_kwargs={"k": 3})

    #print(vectorstore_chroma)
    console.print(f"[green]Documentos {len(documents)} cargados.[/green]")
    
    # inicializamos el modelo de chat 23/01/2024
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, max_tokens=70)
     # Crear la instancia de RetrievalQA fuera de la función
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
       
    ### Marcar mensajes anteriores que llegaron a enviarse por parte de los usuarios
    ### mientras el bot estaba inactivo, para que no sean procesados y optimizar costos.
    mark_new_messages()
    #######
    offset = 0
    while True:
        updates = get_updates(offset)
        if updates:
            for update in updates:
                if 'message' in update and 'chat' in update['message']:
                    user_message = update["message"]["text"]
                    chat_id = update["message"]["chat"]['id']
                    message = update["message"]
                    username = message["from"]["first_name"]
                    print(f"Usuario: {username} ({chat_id})")
                    print(f"Received message: {user_message}")
                    print("---")
                    
                    if user_message.lower() == "/exit":
                        print("Saliendo del programa...")
                        print(update["update_id"])
                        return
                    # funcion para procesar la consulta recibida y retorna respuesta en formato texto
                    response = process_qa_query(query=user_message.lower(), retriever=qa_chain, llm=llm)
                    #Respondemos al mensaje
                    send_messages(chat_id, response)
                    print(offset)
                    offset = update["update_id"] +1
        else:
            time.sleep(1)
if __name__ == '__main__':
    main()
