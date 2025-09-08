# ai_pipeline.py

from glob import glob
import os
import json
from dotenv import load_dotenv

from openai import OpenAI
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from .scrapers import scrape_drugs
from .prompts import (
    hypertention_function,
    system_prompt_tools_llm,
    find_doctors_function,
    system_prompt_doctors_llm,
    SYSTEM_TEMPLATE,
    scrape_drugs_function,
    system_prompt_drugs_llm,
)
from .handle_tools import handle_tool_call, handle_doctors_call

# =========================
# INIT & ENV
# =========================
print("=======================================================================")
print("🚀 [INIT] Starting AI Pipeline Initialization...")
print("=======================================================================")

load_dotenv()
METIS_API_KEY = os.getenv("METIS_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME_LLM = os.getenv("MODEL_NAME_LLM")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME")

openai_client = OpenAI(api_key=METIS_API_KEY, base_url=BASE_URL)

def get_model_params(selected_model: str):
    """
    Returns the parameters needed to initialize an AI client.
    """
    if selected_model == "local_llama":
        print(f"🧠 [CONFIG] Using Local Model: {LOCAL_MODEL_NAME}")
        return {
            "model_name": LOCAL_MODEL_NAME,
            "api_key": "ollama",
            "base_url": OLLAMA_API_URL,
        }
    else:  # Default to cloud GPT
        print(f"🧠 [CONFIG] Using Cloud Model: {MODEL_NAME_LLM}")
        return {
            "model_name": MODEL_NAME_LLM,
            "api_key": METIS_API_KEY,
            "base_url": BASE_URL,
        }

# =========================
# KNOWLEDGE BASE (PDF) — همانند کد ۱
# =========================
try:
    print("📚 [INIT] Loading Knowledge Base documents from PDF files...")
    folders = glob("knowledge_base/*")
    documents = []
    for folder in folders:
        folder_name = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = folder_name
            documents.append(doc)
    print(f"   -> Successfully loaded {len(documents)} documents in total.")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print("🧠 [INIT] Creating embeddings and FAISS vector store for Knowledge Base...")
    model_name_embedding = (
        "/home/amir/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/"
        "snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    )

    kb_embeddings = HuggingFaceEmbeddings(
        model_name=model_name_embedding,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embedding=kb_embeddings)
    print(f"   -> ✅ Knowledge Base vector store created with {vectorstore.index.ntotal} vectors.")
except Exception as e:
    print(f"🔥🔥🔥 [INIT-ERROR] CRITICAL FAILURE during Knowledge Base RAG setup: {e}")
    vectorstore = None  # Ensure vectorstore exists but is None if setup fails

print("=======================================================================")
print("✅ [INIT] AI Pipeline Initialized and Ready.")
print("=======================================================================")

# =======================================================
# CSV VECTORSTORE (Drugstores) — لود تنبل ایمن برای Celery
# =======================================================
_csv_vs = None
_csv_emb = None

def _build_and_save_csv_index(index_path: str = "vectorstores/csv_index"):
    """
    فقط اگر ایندکس وجود نداشت: CSV را لود، تکه‌تکه و ایندکس را می‌سازد و ذخیره می‌کند.
    """
    print("💊 [CSV] Building CSV FAISS index (first-time) ...")
    csv_path = "data_csv/drugstores.csv"
    csv_loader = CSVLoader(file_path=csv_path)
    csv_docs = csv_loader.load()
    for doc in csv_docs:
        doc.metadata["doc_type"] = "csv"

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    csv_chunks = splitter.split_documents(csv_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="/home/amir/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/"
                   "snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vs = FAISS.from_documents(csv_chunks, embedding=embeddings)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    vs.save_local(index_path)
    print(f"   -> ✅ Drugstore vector store created with {vs.index.ntotal} vectors.")
    return vs, embeddings

def get_csv_vectorstore(index_path: str = "vectorstores/csv_index") -> FAISS:
    """
    در هر پردازه یک‌بار لود می‌شود. اگر ایندکس وجود نداشت، ساخته و ذخیره می‌شود.
    """
    global _csv_vs, _csv_emb
    if _csv_emb is None:
        _csv_emb = HuggingFaceEmbeddings(
            model_name="/home/amir/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/"
                       "snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if _csv_vs is None:
        try:
            _csv_vs = FAISS.load_local(
                index_path,
                embeddings=_csv_emb,
                allow_dangerous_deserialization=True,
            )
            print("💊 [CSV] Loaded existing CSV FAISS index.")
        except Exception:
            # اگر ایندکس نبود، بساز
            _csv_vs, _csv_emb = _build_and_save_csv_index(index_path)

    return _csv_vs

# =========================
# Tools & Chains
# =========================
tools_hypertension = [{"type": "function", "function": hypertention_function}]

def chat_with_tools_llm(client: OpenAI, model_name: str, message_text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt_tools_llm},
        {"role": "user", "content": message_text},
    ]
    response = client.chat.completions.create(
        model=model_name, messages=messages, tools=tools_hypertension
    )
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_response = handle_tool_call(message)
        messages.append(message)
        messages.append(tool_response)
        final_response = client.chat.completions.create(model=model_name, messages=messages)
        return final_response.choices[0].message.content
    return "{}"

tools_doctors = [{"type": "function", "function": find_doctors_function}]

def chat_with_doctors_llm(client: OpenAI, model_name: str, message_text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt_doctors_llm},
        {"role": "user", "content": message_text},
    ]
    response = client.chat.completions.create(
        model=model_name, messages=messages, tools=tools_doctors
    )
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_response = handle_doctors_call(message)
        messages.append(message)
        messages.append(tool_response)
        final_response = client.chat.completions.create(model=model_name, messages=messages)
        return final_response.choices[0].message.content
    return ""

tools_drugs = [{"type": "function", "function": scrape_drugs_function}]

def chat_with_drugs_llm(llm: ChatOpenAI, user_message: str) -> str:
    """
    Agent برای دارو/داروخانه‌ها.
    - ابزار scrape_drugs: اسکرپ قیمت/لینک
    - ابزار csv_retriever: صرفاً similarity-search روی ایندکس CSV (بدون LLM داخل Tool)
    """
    # سیستم‌پرومپت
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
        - Search the drugstore CSV to find nearby drugstores.
        - Return full drugstore info + scraped drug info in JSON.
    3. If no drug is found by scrape_drugs → return nothing.
    All responses must be in JSON only.
    """

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # ---- Tool 1: scrape_drugs (مثل کد ۱)
    def scrape_tool_func(medicine_name: str) -> str:
        result = scrape_drugs(medicine_name)
        return result

    desc_drug_tool = (
        "Fetch product details for a given supplement or drug name from mokamelkhoone.com. "
        "Returns the title, current price, product info, and a direct URL. "
        "Use this when a user wants to look up the price or details of a supplement product."
    )
    scrape_tool = Tool(name="scrape_drugs", func=scrape_tool_func, description=desc_drug_tool)

    # ---- Tool 2: csv_retriever (بدون LLM-call داخل Tool)
    def rag_tool_func(drug_name: str) -> str:
        # صرفاً بازیابی شباهتی از FAISS، سریع و بدون LLM
        vs = get_csv_vectorstore()
        retriever = vs.as_retriever(search_kwargs={"k": 10})
        docs = retriever.get_relevant_documents(drug_name)

        def doc_to_row(d):
            meta = getattr(d, "metadata", {}) or {}
            return {
                "title": meta.get("source") or meta.get("file") or "drugstore",
                "address": meta.get("address") or meta.get("Address") or "",
                "phone": meta.get("phone") or meta.get("Phone") or "",
                "snippet": d.page_content[:200],
            }

        rows = [doc_to_row(d) for d in docs]
        out = {
            "nearby_drugstores": rows,
            "note": "نتایج بر اساس شباهت از CSV (بدون تضمین موجودی).",
        }
        return json.dumps(out, ensure_ascii=False)

    rag_tool = Tool(
        name="csv_retriever",
        func=rag_tool_func,
        description=(
            "Searches drugstore CSV to locate nearby drugstores. "
            "Uses similarity search without LLM inside the tool."
        ),
    )

    agent = initialize_agent(
        tools=[scrape_tool, rag_tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        agent_kwargs={"system_message": system_prompt},
        verbose=True,
    )

    return agent.run(user_message)

# =========================
# RAG QA Prompt (مثل کد ۱)
# =========================
QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=SYSTEM_TEMPLATE)

def chat_with_rag_llm(llm: ChatOpenAI, tools_response, doctors_response, drugs_response: str) -> str:
    if not vectorstore:
        return "خطا: پایگاه دانش به درستی بارگذاری نشده است. امکان تولید گزارش کامل وجود ندارد."

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever()

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, memory=memory, retriever=retriever, combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    question_input = (
        "🔸 Disease Prediction Output:\n"
        + json.dumps(tools_response, ensure_ascii=False, indent=2)
        + "\n\n🔸 List of Doctors:\n"
        + json.dumps(doctors_response, ensure_ascii=False, indent=2)
        + "\n\n🔸 Proper Medicine For The Patient:\n"
        + drugs_response
    )
    result = conversation_chain.invoke({"question": question_input})
    return result["answer"]

# =========================
# Entry Point (مثل کد ۱) + timeouts
# =========================
def run_health_analysis_pipeline(profile_text_summary: str, selected_model: str = "cloud_gpt") -> str:
    # 1) تنظیمات مدل
    params = get_model_params(selected_model)

    # 2) کلاینت OpenAI (برای ابزارهای فانکشن)
    client = OpenAI(api_key=params["api_key"], base_url=params["base_url"])

    # 3) LLMها با timeout و retry محدود (برای جلوگیری از گیر)
    agent_llm = ChatOpenAI(
        base_url=params["base_url"],
        api_key=params["api_key"],
        temperature=0.2,
        model_name=params["model_name"],
        timeout=30,
        max_retries=1,
    )
    rag_llm = ChatOpenAI(
        base_url=params["base_url"],
        api_key=params["api_key"],
        temperature=0.8,
        model_name=params["model_name"],
        timeout=30,
        max_retries=1,
    )

    # --
    print("\n\n--- [START] New Health Analysis Request ---")

    # STAGE 1: Disease Prediction
    print("\n[PIPELINE - STAGE 1] 🩺 Running Disease Prediction...")
    tools_response_str = chat_with_tools_llm(client, params["model_name"], profile_text_summary)
    try:
        print(tools_response_str)
        tools_response_json = json.loads(tools_response_str)
        if not tools_response_json:
            raise json.JSONDecodeError("Empty JSON object returned", tools_response_str, 0)
        print("   -> ✅ Success")
    except json.JSONDecodeError:
        print("   -> ❌ FAILED. Model did not return valid JSON for disease prediction.")
        print(f"   -> Raw Output: {tools_response_str}")
        print("--- [END] Request Failed ---")
        return "گزارش به دلیل عدم تشخیص وضعیت پزشکی توسط مدل اول، تولید نشد."

    # STAGE 2: Doctors
    print("\n[PIPELINE - STAGE 2] 👨‍⚕️ Finding Relevant Doctors...")
    doctors_response_str = chat_with_doctors_llm(client, params["model_name"], tools_response_str)
    if not doctors_response_str or not doctors_response_str.strip().startswith("["):
        print("   -> 🟡 Note: No doctors were found for the given criteria.")
        doctors_response_str = "[]"
    else:
        print("   -> ✅ Success.")

    # STAGE 3: Drugs / Drugstores
    print("\n[PIPELINE - STAGE 3] 💊 Searching for Drugs and Drugstores...")
    drugs_response = chat_with_drugs_llm(agent_llm, tools_response_str)
    print("   -> ✅ Success.")

    # STAGE 4: Final Report via RAG
    print("\n[PIPELINE - STAGE 4] 📄 Generating Final Report with RAG...")
    final_report = chat_with_rag_llm(
        rag_llm, tools_response_json, json.loads(doctors_response_str), drugs_response
    )
    print("   -> ✅ Success.")

    print("\n--- [END] Health Analysis Request Finished Successfully ---")
    return final_report
