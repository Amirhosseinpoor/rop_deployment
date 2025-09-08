
from glob import glob
from langchain.document_loaders.csv_loader import CSVLoader
from openai import OpenAI
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import json


from .disease_models import predict_hypertension_risk
from .scrapers import scrape_doctors, scrape_drugs
from .prompts import hypertention_function, system_prompt_tools_llm, find_doctors_function, system_prompt_doctors_llm, SYSTEM_TEMPLATE, scrape_drugs_function,system_prompt_drugs_llm
from.handle_tools import handle_tool_call, handle_doctors_call

print("=======================================================================")
print("🚀 [INIT] Starting AI Pipeline Initialization...")
print("=======================================================================")

load_dotenv()
METIS_API_KEY = os.getenv('METIS_API_KEY')
BASE_URL = os.getenv('BASE_URL')
MODEL_NAME_LLM = os.getenv('MODEL_NAME_LLM')
openai_client = OpenAI(api_key=METIS_API_KEY, base_url=BASE_URL)
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL')
LOCAL_MODEL_NAME = os.getenv('LOCAL_MODEL_NAME')

def get_model_params(selected_model):
    """
    Returns the parameters needed to initialize an AI client.
    """
    if selected_model == 'local_llama':
        print(f"🧠 [CONFIG] Using Local Model: {LOCAL_MODEL_NAME}")
        return {
            "model_name": LOCAL_MODEL_NAME,
            "api_key": "ollama",
            "base_url": OLLAMA_API_URL
        }
    else:  # Default to cloud GPT
        print(f"🧠 [CONFIG] Using Cloud Model: {MODEL_NAME_LLM}")
        return {
            "model_name": MODEL_NAME_LLM,
            "api_key": METIS_API_KEY,
            "base_url": BASE_URL
        }
try:
    print("📚 [INIT] Loading Knowledge Base documents from PDF files...")
    folders = glob("knowledge_base/*")
    documents = []
    for folder in folders:
        folder_name = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = folder_name
            documents.append(doc)
    print(f"   -> Successfully loaded {len(documents)} documents in total.")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print("🧠 [INIT] Creating embeddings and FAISS vector store for Knowledge Base...")
    model_name_embedding = "/home/amir/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name_embedding,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    print(f"   -> ✅ Knowledge Base vector store created with {vectorstore.index.ntotal} vectors.")

except Exception as e:
    print(f"🔥🔥🔥 [INIT-ERROR] CRITICAL FAILURE during Knowledge Base RAG setup: {e}")
    vectorstore = None  # Ensure vectorstore exists but is None if setup fails

print("💊 [INIT] Loading Drugstore data from CSV file...")
csv_path = "data_csv/drugstores.csv"
csv_loader = CSVLoader(file_path=csv_path)
csv_docs = csv_loader.load()
for doc in csv_docs:
    doc.metadata["doc_type"] = "csv"

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
csv_chunks = text_splitter.split_documents(csv_docs)

csv_embeddings = HuggingFaceEmbeddings(
    model_name="/home/amir/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",

    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

csv_vectorstore = FAISS.from_documents(csv_chunks, embedding=csv_embeddings)
csv_vectorstore.save_local("vectorstores/csv_index")
print(f"   -> ✅ Drugstore vector store created with {csv_vectorstore.index.ntotal} vectors.")

print("=======================================================================")
print("✅ [INIT] AI Pipeline Initialized and Ready.")
print("=======================================================================")


tools_hypertension = [{"type": "function", "function": hypertention_function}]
def chat_with_tools_llm(client, model_name, message_text):
    messages = [{"role": "system", "content": system_prompt_tools_llm}, {"role": "user", "content": message_text}]
    response = client.chat.completions.create(model=model_name, messages=messages, tools=tools_hypertension)
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_response = handle_tool_call(message)
        messages.append(message)
        messages.append(tool_response)
        final_response = client.chat.completions.create(model=model_name, messages=messages)
        return final_response.choices[0].message.content
    return "{}"

tools_doctors = [{"type": "function", "function": find_doctors_function}]

def chat_with_doctors_llm(client, model_name, message_text):
    messages = [{"role": "system", "content": system_prompt_doctors_llm}, {"role": "user", "content": message_text}]
    response = client.chat.completions.create(model=model_name, messages=messages, tools=tools_doctors)
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_response = handle_doctors_call(message)
        messages.append(message)
        messages.append(tool_response)
        final_response = client.chat.completions.create(model=model_name, messages=messages)
        return final_response.choices[0].message.content
    return ""

tools_drugs = [{"type": "function", "function": scrape_drugs_function}]

def chat_with_drugs_llm(llm, user_message):

    csv_retriever = csv_vectorstore.as_retriever()


    system_prompt = """
    You are a smart medicine assistant.

    If the user mentions a diagnosis like:
    - "⚠️ Based on the model, the patient has a 74.5% probability of having hypertension."
    Then assume the user needs a supplement.

    Diagnosis → Recommended Drug:
    - Hypertension → Magnesium citrate

    Steps:
    1. Call the scrape_drugs tool with the drug name.
    2. If scrape_drugs returns data:
        - Search the drugstore CSV to find a matching drugstore with that drug.
        - If found → return full drugstore + drug info in JSON.
        - If not found → return nearest drugstore info + drug info + note.
    3. If no drug is found by scrape_drugs → return nothing.
    All responses must be in JSON only.
    """

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def scrape_tool_func(medicine_name: str) -> str:
        result = scrape_drugs(medicine_name)
        return result

    desc_drug_tool = '''Fetch product details for a given supplement or drug name from mokamelkhoone.com. 
    Returns the title, current price, product info, and a direct URL. 
    Use this when a user wants to look up the price or details of a supplement product.
    '''
    scrape_tool = Tool(
        name="scrape_drugs",
        func=scrape_tool_func,
        description=desc_drug_tool
    )

    def rag_tool_func(drug_name: str) -> str:
        query_primary = f"Find a drugstore near to the region of patient that has '{drug_name}' in stock. For example, if the region was vanak you should search in vanak square at address section in CSV."
        result = RetrievalQA.from_chain_type(llm=llm, retriever=csv_retriever).run(query_primary)

        fallback_query = "List the nearest drugstores located in patient's region regardless of drug availability. Return their address and contact details so the patient can visit them in person to ask about the drug. Pay attention to address in CSV not to province or city."
        fallback_result = RetrievalQA.from_chain_type(llm=llm, retriever=csv_retriever).run(fallback_query)

        if "not found" not in result.lower() and "unavailable" not in result.lower():
            combined = {
                "drugstore_with_drug": result,
                "nearby_drugstores": fallback_result
            }
        else:
            combined = {
                "note": "The drug was not found in any nearby drugstores. However, here are some drugstores in the patient’s region that they can visit and ask in person.",
                "nearby_drugstores": fallback_result
            }

        return json.dumps(combined, ensure_ascii=False)

    rag_tool = Tool(
        name="csv_retriever",
        func=rag_tool_func,
        description="Searches drugstore CSV to locate a drug and its availability details. You should pay attention to address, not in province and city"
    )

    agent = initialize_agent(
        tools=[scrape_tool, rag_tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        agent_kwargs={"system_message": system_prompt},
        verbose=True
    )

    return agent.run(user_message)


QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=SYSTEM_TEMPLATE)


def chat_with_rag_llm(llm, tools_response, doctors_response, drugs_response):
    if not vectorstore:
        return "خطا: پایگاه دانش به درستی بارگذاری نشده است. امکان تولید گزارش کامل وجود ندارد."


    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever()

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, memory=memory, retriever=retriever, combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    question_input = (
            "🔸 Disease Prediction Output:\n" +
            json.dumps(tools_response, ensure_ascii=False, indent=2) +
            "\n\n🔸 List of Doctors:\n" +
            json.dumps(doctors_response, ensure_ascii=False, indent=2) +
            "\n\n🔸 Proper Medicine For The Patient:\n" +
            drugs_response
    )
    result = conversation_chain.invoke({"question": question_input})
    return result["answer"]


def run_health_analysis_pipeline(profile_text_summary, selected_model='cloud_gpt'):
    # --- EFFICIENCY IMPROVEMENT ---
    # 1. Get model parameters once.
    params = get_model_params(selected_model)

    # 2. Create client instances once.
    client = OpenAI(api_key=params["api_key"], base_url=params["base_url"])

    # Create LangChain-compatible LLM objects for the agent-based chains
    agent_llm = ChatOpenAI(
        base_url=params["base_url"], api_key=params["api_key"],
        temperature=0.2, model_name=params["model_name"]
    )
    rag_llm = ChatOpenAI(
        base_url=params["base_url"], api_key=params["api_key"],
        temperature=0.8, model_name=params["model_name"]
    )
    # --
    print("\n\n--- [START] New Health Analysis Request ---")
    print("\n[PIPELINE - STAGE 1] 🩺 Running Disease Prediction...")
    tools_response_str = chat_with_tools_llm(client, params["model_name"], profile_text_summary)
    try:
        tools_response_json = json.loads(tools_response_str)
        if not tools_response_json:
            raise json.JSONDecodeError("Empty JSON object returned", tools_response_str, 0)
        print(f"   -> ✅ Success")
    except json.JSONDecodeError:
        print(f"   -> ❌ FAILED. Model did not return valid JSON for disease prediction.")
        print(f"   -> Raw Output: {tools_response_str}")
        print("--- [END] Request Failed ---")
        return "گزارش به دلیل عدم تشخیص وضعیت پزشکی توسط مدل اول، تولید نشد."

    print("\n[PIPELINE - STAGE 2] 👨‍⚕️ Finding Relevant Doctors...")
    doctors_response_str = chat_with_doctors_llm(client, params["model_name"], tools_response_str)

    if not doctors_response_str or not doctors_response_str.strip().startswith('['):
        print("   -> 🟡 Note: No doctors were found for the given criteria.")
        doctors_response_str = "[]"
    else:
        print(f"   -> ✅ Success.")

    print("\n[PIPELINE - STAGE 3] 💊 Searching for Drugs and Drugstores...")
    drugs_response = chat_with_drugs_llm(agent_llm, tools_response_str)
    print(f"   -> ✅ Success.")

    print("\n[PIPELINE - STAGE 4] 📄 Generating Final Report with RAG...")
    final_report = chat_with_rag_llm(rag_llm, tools_response_json, json.loads(doctors_response_str), drugs_response)
    print("   -> ✅ Success.")

    print("\n--- [END] Health Analysis Request Finished Successfully ---")
    return final_report
