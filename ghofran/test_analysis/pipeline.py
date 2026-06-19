"""Multi-stage health-analysis report pipeline.

Ported from the original ``ai_pipeline.py``. The four stages are unchanged:

1. **Disease prediction** — an LLM extracts vitals and calls the hypertension
   model tool, returning a JSON verdict.
2. **Doctor finding** — an LLM calls ``scrape_doctors`` when hypertension is found.
3. **Drug & pharmacy search** — a LangChain agent looks up a supplement and
   matches it against the drugstore CSV (RAG).
4. **Report generation** — a ConversationalRetrievalChain over the PDF knowledge
   base writes the final Persian report.

Operational changes: all model ids / keys / paths come from :mod:`config`; the
embedding-model path and corpus/CSV locations are configurable; the two heavy
vectorstores are built lazily and cached per process (as in the original).
"""
from __future__ import annotations

import json
from glob import glob
import os

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from openai import OpenAI

from .config import get_settings
from .handle_tools import handle_doctors_call, handle_tool_call
from .prompts import (
    SYSTEM_TEMPLATE,
    find_doctors_function,
    hypertention_function,
    scrape_drugs_function,
    system_prompt_doctors_llm,
    system_prompt_drugs_llm,
    system_prompt_tools_llm,
)
from .scrapers import scrape_drugs

# Per-process vectorstore cache (built lazily on first report request).
_resources: dict = {}


def _get_model_params(selected_model: str) -> dict:
    """Return ``{model_name, api_key, base_url}`` for the chosen backend."""
    settings = get_settings()
    if selected_model == "local_llama":
        return {"model_name": settings.local_model_name, "api_key": "ollama", "base_url": settings.ollama_api_url}
    return {"model_name": settings.pipeline_model, "api_key": settings.pipeline_api_key, "base_url": settings.pipeline_base_url}


def _embeddings() -> HuggingFaceEmbeddings:
    """Construct the configured sentence-embedding model (CPU, normalised)."""
    return HuggingFaceEmbeddings(
        model_name=get_settings().embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore():
    """Lazily build & cache the PDF knowledge-base FAISS store."""
    if "vectorstore" not in _resources:
        try:
            kb_dir = get_settings().knowledge_base_dir
            documents = []
            for folder in glob(os.path.join(kb_dir, "*")):
                folder_name = os.path.basename(folder)
                loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
                for doc in loader.load():
                    doc.metadata["doc_type"] = folder_name
                    documents.append(doc)
            chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
            _resources["vectorstore"] = FAISS.from_documents(chunks, embedding=_embeddings())
        except Exception as e:  # noqa: BLE001
            print(f"🔥 [INIT-ERROR] Knowledge Base RAG setup failed: {e}")
            _resources["vectorstore"] = None
    return _resources["vectorstore"]


def get_csv_vectorstore():
    """Lazily build & cache the drugstore-CSV FAISS store."""
    if "csv_vectorstore" not in _resources:
        try:
            loader = CSVLoader(file_path=get_settings().drugstores_csv)
            docs = loader.load()
            for doc in docs:
                doc.metadata["doc_type"] = "csv"
            chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
            _resources["csv_vectorstore"] = FAISS.from_documents(chunks, embedding=_embeddings())
        except Exception as e:  # noqa: BLE001
            print(f"🔥 [INIT-ERROR] CSV RAG setup failed: {e}")
            _resources["csv_vectorstore"] = None
    return _resources["csv_vectorstore"]


# --- Stage 1: disease prediction --------------------------------------------
_tools_hypertension = [{"type": "function", "function": hypertention_function}]


def chat_with_tools_llm(client, model_name, message_text) -> str:
    """Stage 1: extract vitals, call the hypertension tool, return JSON verdict."""
    messages = [
        {"role": "system", "content": system_prompt_tools_llm},
        {"role": "user", "content": message_text},
    ]
    response = client.chat.completions.create(model=model_name, messages=messages, tools=_tools_hypertension)
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_response = handle_tool_call(message)
        messages.extend([message, tool_response])
        final = client.chat.completions.create(model=model_name, messages=messages)
        return final.choices[0].message.content
    return "{}"


# --- Stage 2: doctor finding ------------------------------------------------
_tools_doctors = [{"type": "function", "function": find_doctors_function}]


def chat_with_doctors_llm(client, model_name, message_text) -> str:
    """Stage 2: when hypertension is found, call ``scrape_doctors`` and return JSON."""
    messages = [
        {"role": "system", "content": system_prompt_doctors_llm},
        {"role": "user", "content": message_text},
    ]
    response = client.chat.completions.create(model=model_name, messages=messages, tools=_tools_doctors)
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_response = handle_doctors_call(message)
        messages.extend([message, tool_response])
        final = client.chat.completions.create(model=model_name, messages=messages)
        return final.choices[0].message.content
    return ""


# --- Stage 3: drug & pharmacy search ----------------------------------------
def chat_with_drugs_llm(llm, user_message) -> str:
    """Stage 3: a LangChain agent finds a supplement and matches a drugstore (RAG)."""
    csv_vectorstore = get_csv_vectorstore()
    if not csv_vectorstore:
        return json.dumps({"error": "Drugstore database could not be loaded."})
    csv_retriever = csv_vectorstore.as_retriever()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def scrape_tool_func(medicine_name: str) -> str:
        return scrape_drugs(medicine_name)

    scrape_tool = Tool(
        name="scrape_drugs",
        func=scrape_tool_func,
        description=(
            "Fetch product details for a given supplement/drug name from mokamelkhoone.com. "
            "Returns title, price, info and a URL."
        ),
    )

    def rag_tool_func(drug_name: str) -> str:
        primary = RetrievalQA.from_chain_type(llm=llm, retriever=csv_retriever).run(
            f"Find a drugstore near the patient's region that has '{drug_name}' in stock. "
            "Search the address section in the CSV."
        )
        fallback = RetrievalQA.from_chain_type(llm=llm, retriever=csv_retriever).run(
            "List the nearest drugstores in the patient's region regardless of drug availability. "
            "Return address and contact details. Pay attention to the address column in the CSV."
        )
        if "not found" not in primary.lower() and "unavailable" not in primary.lower():
            combined = {"drugstore_with_drug": primary, "nearby_drugstores": fallback}
        else:
            combined = {
                "note": "The drug was not found nearby; here are drugstores in the patient's region to ask in person.",
                "nearby_drugstores": fallback,
            }
        return json.dumps(combined, ensure_ascii=False)

    rag_tool = Tool(
        name="csv_retriever",
        func=rag_tool_func,
        description="Searches the drugstore CSV to locate a drug and its availability (focus on address).",
    )

    agent = initialize_agent(
        tools=[scrape_tool, rag_tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        agent_kwargs={"system_message": system_prompt_drugs_llm},
        verbose=False,
    )
    return agent.run(user_message)


# --- Stage 4: final report --------------------------------------------------
_QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=SYSTEM_TEMPLATE)


def chat_with_rag_llm(llm, tools_response, doctors_response, drugs_response) -> str:
    """Stage 4: generate the final Persian report grounded in the PDF knowledge base."""
    vectorstore = get_vectorstore()
    if not vectorstore:
        return "خطا: پایگاه دانش به درستی بارگذاری نشده است. امکان تولید گزارش کامل وجود ندارد."

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, memory=memory, retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": _QA_PROMPT},
    )
    question_input = (
        "🔸 Disease Prediction Output:\n" + json.dumps(tools_response, ensure_ascii=False, indent=2) +
        "\n\n🔸 List of Doctors:\n" + json.dumps(doctors_response, ensure_ascii=False, indent=2) +
        "\n\n🔸 Proper Medicine For The Patient:\n" + drugs_response
    )
    return chain.invoke({"question": question_input})["answer"]


def run_health_analysis_pipeline(profile_text_summary: str, selected_model: str = "cloud_gpt") -> str:
    """Run the full 4-stage pipeline and return the final Persian report text.

    Args:
        profile_text_summary: Free-text summary of the patient's profile/vitals.
        selected_model: ``"cloud_gpt"`` (default, Metis) or ``"local_llama"`` (Ollama).
    """
    params = _get_model_params(selected_model)
    client = OpenAI(api_key=params["api_key"], base_url=params["base_url"])
    agent_llm = ChatOpenAI(base_url=params["base_url"], api_key=params["api_key"],
                           temperature=0.2, model_name=params["model_name"])
    rag_llm = ChatOpenAI(base_url=params["base_url"], api_key=params["api_key"],
                         temperature=0.8, model_name=params["model_name"])

    # Stage 1
    tools_response_str = chat_with_tools_llm(client, params["model_name"], profile_text_summary)
    try:
        tools_response_json = json.loads(tools_response_str)
        if not tools_response_json:
            raise json.JSONDecodeError("Empty JSON object returned", tools_response_str, 0)
    except json.JSONDecodeError:
        return "گزارش به دلیل عدم تشخیص وضعیت پزشکی توسط مدل اول، تولید نشد."

    # Stage 2
    doctors_response_str = chat_with_doctors_llm(client, params["model_name"], tools_response_str)
    if not doctors_response_str or not doctors_response_str.strip().startswith("["):
        doctors_response_str = "[]"

    # Stage 3
    drugs_response = chat_with_drugs_llm(agent_llm, tools_response_str)

    # Stage 4
    return chat_with_rag_llm(rag_llm, tools_response_json, json.loads(doctors_response_str), drugs_response)
