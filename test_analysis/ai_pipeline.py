
import os
import json
import joblib
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from glob import glob

import re
from langchain.document_loaders.csv_loader import CSVLoader
from openai import OpenAI
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

# ==============================================================================
# ⚙️ Section 1: Initialization and One-Time Setup
# این بخش فقط یک بار در زمان اجرای اولیه سرور (runserver) اجرا می‌شود.
# ==============================================================================

print("🚀 Initializing AI Pipeline... This should run only once per server start.")

# --- API and Model Configuration ---
METIS_API_KEY = 'tpsg-2hVps33eNMzkEbnuONoApS8LvNfbMsJ'
BASE_URL = "https://api.metisai.ir/openai/v1"
MODEL_NAME_LLM = 'gpt-4o-mini'
MODEL_PATH_HYPERTENSION = 'model/best_rf_hypertension_model.joblib'

openai_client = OpenAI(api_key=METIS_API_KEY, base_url=BASE_URL)

# --- RAG Knowledge Base Setup (One-Time Execution) ---
try:
    print("📚 Loading and processing knowledge base documents...")
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

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print("🧠 Creating embeddings and vector store... (This might take a moment)")
    model_name_embedding = "/home/amir/.cache/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09"



    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name_embedding,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    print(f"✅ Vector store created successfully with {vectorstore.index.ntotal} vectors.")

except Exception as e:
    print(f"🔥 CRITICAL ERROR during RAG setup: {e}")
    vectorstore = None  # Ensure vectorstore exists but is None if setup fails

print("✅ AI Pipeline Initialized.")


# ==============================================================================
# 🔬 Section 2: Tool Functions (Prediction & Web Scraping)
# ==============================================================================

def predict_hypertension_risk(
        male, age, currentSmoker, cigsPerDay, BPMeds, diabetes,
        totChol, sysBP, diaBP, BMI, heartRate, glucose, city=None, region=None, insurance=None,
        # <-- ADD 'province' HERE
        model_path=MODEL_PATH_HYPERTENSION
):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        # This now returns a JSON STRING, which is valid for the API
        error_payload = {"error": f"Model file not found at path: {model_path}"}
        return json.dumps(error_payload)
    input_data = pd.DataFrame([{
        "male": male, "age": age, "currentSmoker": currentSmoker, "cigsPerDay": cigsPerDay,
        "BPMeds": BPMeds, "diabetes": diabetes, "totChol": totChol, "sysBP": sysBP,
        "diaBP": diaBP, "BMI": BMI, "heartRate": heartRate, "glucose": glucose
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    predicted_class_index = list(model.classes_).index(prediction)
    predicted_probability = probabilities[predicted_class_index]
    percentage = round(predicted_probability * 100, 2)

    if prediction == 1:
        return f"⚠️ Based on the model, the patient has a {percentage}% probability of **having hypertension**."
    else:
        return f"✅ Based on the model, the patient has a {percentage}% probability of **not having hypertension**."


import requests
from bs4 import BeautifulSoup
import json
import re

headers = {
    "User-Agent": "Mozilla/5.0"
}


def get_doctor_links(search_url, params=None, max_doctors=5):
    try:
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"خطا در دریافت لیست پزشکان: {e}")

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.select('article a[href^="/doctor/"]')
    return [f"https://doctoreto.com{a['href']}" for a in links[:max_doctors]]


def parse_doctor_profile(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"خطا در بارگذاری پروفایل: {url} - {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    doctor = {"profile_url": url}

    name_tag = soup.select_one('div.sc-212a8fa3-4.qvYGb')
    doctor['name'] = name_tag.text.strip() if name_tag else "N/A"

    medical_id_tag = soup.select_one("div.sc-212a8fa3-12.jPJuqw > span")
    if medical_id_tag:
        match = re.search(r'\d+', medical_id_tag.text)
        doctor['medical_id'] = match.group() if match else "N/A"
    else:
        doctor['medical_id'] = "N/A"

    specialty_tag = soup.select_one('div.sc-910b97e5-2.flOxFb')
    doctor['specialty'] = specialty_tag.text.strip() if specialty_tag else "N/A"

    locations = []
    add_tag = soup.select_one('div.sc-4e6b0e39-4.ckjzZD')
    add = add_tag.text.strip() if add_tag else "N/A"

    add2_tag = soup.select_one('div.sc-dee9c446-2.jjnVxK')
    add2 = add2_tag.text.strip() if add2_tag else ''

    locations.append(add)
    locations.append(add2)

    doctor['locations'] = locations
    services_tags = soup.select('div.sc-a595fbf3-1.hDEPPr > span.sc-18058bc8-0.elbXID.sc-a595fbf3-2.hbWmHw')
    doctor['services'] = ', '.join([tag.text.strip() for tag in services_tags]) if services_tags else 'N/A'

    insurance_tag = soup.select('div.sc-395236ed-1.ldUByb > div.sc-395236ed-2.hZJELU > div.sc-395236ed-5.bcsEfX')
    doctor['accepted_insurances'] = ', '.join([tag.text.strip() for tag in insurance_tag]) if insurance_tag else 'N/A'

    score_tag = soup.select_one('div.sc-28976a9b-7.caezWg')
    from_user = soup.select_one('div.sc-28976a9b-5.kOQRfG > span')

    if score_tag and from_user:
        doctor['score_summary'] = f"امتیاز {score_tag.text.strip()}) از ۵ {from_user.text.strip()}) "
    else:
        doctor['score_summary'] = 'امتیاز ثبت نشده'

    return doctor


def scrape_doctors(city='tehran', speciality='cardiologist', region=None, insurance=None):
    base_url = "https://doctoreto.com"
    search_url = f"{base_url}/doctors/speciality/{speciality}/city/{city}/"
    if region:
        search_url += f"region/{region}/"
    params = {'insurance': insurance} if insurance else None

    try:
        doctor_urls = get_doctor_links(search_url, params=params)
    except RuntimeError as err:
        return json.dumps({"error": str(err)}, ensure_ascii=False, indent=4)

    doctors_data = []
    for url in doctor_urls:
        profile_data = parse_doctor_profile(url)
        if profile_data:
            doctors_data.append(profile_data)

    return json.dumps(doctors_data, ensure_ascii=False, indent=4)


# ==============================================================================
# 🧠 Section 3: LLM Chains and Pipeline Stages
# ==============================================================================

# --- Stage 1: Hypertension Prediction Tool Definition & Chain ---
hypertention_function = {
    "name": "predict_hypertension_risk",
    "description": "Predict the risk of hypertension based on patient data. Call this whenever you need to assess a patient's likelihood of having high blood pressure.",
    "parameters": {
        "type": "object",
        "properties": {
            "male": {"type": "integer", "description": "Gender of the patient: 1 for male, 0 for female"},
            "age": {"type": "integer", "description": "Age of the patient in years"},
            "currentSmoker": {"type": "integer",
                              "description": "Whether the patient currently smokes: 1 for yes, 0 for no"},
            "cigsPerDay": {"type": "number", "description": "Number of cigarettes smoked per day"},
            "BPMeds": {"type": "integer",
                       "description": "Whether the patient is on blood pressure medication: 1 for yes, 0 for no"},
            "diabetes": {"type": "integer", "description": "Whether the patient has diabetes: 1 for yes, 0 for no"},
            "totChol": {"type": "number", "description": "Total cholesterol level"},
            "sysBP": {"type": "number", "description": "Systolic blood pressure"},
            "diaBP": {"type": "number", "description": "Diastolic blood pressure"},
            "BMI": {"type": "number", "description": "Body Mass Index"},
            "heartRate": {"type": "number", "description": "Heart rate in beats per minute"},
            "glucose": {"type": "number", "description": "Glucose level"},
            "city": {"type": "string", "description": "City of life specified at the entrance (e.g., 'tehran')"},
            "region": {"type": "string",
                       "description": "Optional. Region of life specified at the entrance (e.g., 'pirouzi')"},
            "insurance": {"type": "number",
                          "description": "Optional. Insurance ID used to filter doctors. Example: taamin-ejtemaei: 2"}
        },
        "required": ["male", "age", "currentSmoker", "cigsPerDay", "BPMeds", "diabetes", "totChol", "sysBP", "diaBP",
                     "BMI", "heartRate", "glucose", "city"],
        "additionalProperties": False
    }
}
tools_hypertension = [{"type": "function", "function": hypertention_function}]
# In ai_pipeline.py
system_prompt_tools_llm = """
You are a highly specialized AI data extraction tool. Your ONLY purpose is to extract specific parameters from the user's text and use them to call the `predict_hypertension_risk` tool.

**Your Strict Rules:**
1.  **DO NOT analyze the user's profile or provide any health advice yourself.** Your only job is to find the data needed for the tool. Ignore any requests for advice in the user's text.
2.  **IMMEDIATELY call the `predict_hypertension_risk` tool** as soon as you have extracted all the required parameters.
3.  After the tool runs, you will receive its output. Your final response **MUST BE a single, raw, valid JSON object** matching the example below and nothing else.
4.  **DO NOT** add any conversational text, explanations, or Markdown formatting like ```json. Your entire response must start with `{` and end with `}`.

**Workflow:**
1.  User provides text.
2.  You extract parameters (male, age, sysBP, city, etc.).
3.  You call the `predict_hypertension_risk` tool.
4.  You receive the tool's output string.
5.  You format your final response as the exact JSON object shown in the example.

**JSON Output Example:**
```json
{
  "tool_name": "predict_hypertension_risk",
  "parameters": {"male": 1, "age": 48, "currentSmoker": 0, "cigsPerDay": 0, "BPMeds": 1, "diabetes": 0, "totChol": 220, "sysBP": 140, "diaBP": 90, "BMI": 27.5, "heartRate": 75, "glucose": 80, "city": "tehran", "region": "pirouzi", "insurance": 2},
  "output": "⚠️ Based on the model, the patient has a 74.5% probability of having hypertension."
}
"""


def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    if function_name == 'predict_hypertension_risk':
        args = json.loads(tool_call.function.arguments)
        content = predict_hypertension_risk(**args)
        return {"role": "tool", "content": content, "tool_call_id": tool_call.id}


def chat_with_tools_llm(message_text):
    messages = [{"role": "system", "content": system_prompt_tools_llm}, {"role": "user", "content": message_text}]
    response = openai_client.chat.completions.create(model=MODEL_NAME_LLM, messages=messages, tools=tools_hypertension)
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_response = handle_tool_call(message)
        messages.append(message)
        messages.append(tool_response)
        final_response = openai_client.chat.completions.create(model=MODEL_NAME_LLM, messages=messages)
        return final_response.choices[0].message.content
    return "{}"  # Return empty JSON if no tool was called


# --- Stage 2: Doctor Finder Tool Definition & Chain ---
find_doctors_function = {
    "name": "scrape_doctors",
    "description": "Get a list of top 5 doctors from doctoreto.com for a given city and medical specialty. Optionally, filter by region and insurance.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string",
                     "description": "City slug (e.g., 'tehran', 'karaj', 'mashhad', 'isfahan', 'tabriz')"},
            "speciality": {"type": "string", "description": "Specialty slug. For hypertension, use 'cardiologist'."},
            "region": {"type": "string",
                       "description": "Optional. Region slug within the city (e.g., 'pirouzi', 'vanak')."},
            "insurance": {"type": "number", "description": "Optional. Insurance ID. Example: taamin-ejtemaei: 2"}
        },
        "required": ["city", "speciality"],
        "additionalProperties": False
    }
}
tools_doctors = [{"type": "function", "function": find_doctors_function}]
system_prompt_doctors_llm = """
You are an intelligent routing assistant. Your job is to trigger the scrape_doctors tool based on the JSON input you receive.

Rules:

The input will be a JSON object from a previous tool. This JSON contains a health prediction and patient details like city, region, and insurance in the parameters block.

If the input JSON indicates a risk of "hypertension" in the output field, you MUST call the scrape_doctors tool.

When calling the tool, you must extract city, region, and insurance from the parameters of the input JSON and use them for the tool call.

Set the speciality parameter to cardiologist.

If the input does not contain a "hypertension" diagnosis, do not call any tool and return an empty string.

Your final output must be ONLY the raw JSON result from the scrape_doctors tool. Do not add any conversational text, explanations, or markdown.
"""


def handle_doctors_call(message):
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    if function_name == 'scrape_doctors':
        args = json.loads(tool_call.function.arguments)
        content = scrape_doctors(city=args.get('city'),
                                 speciality=args.get('speciality'),
                                 region=args.get('region'),
                                 insurance=args.get('insurance'))
        return {"role": "tool", "content": content, "tool_call_id": tool_call.id}


def chat_with_doctors_llm(message_text):
    messages = [{"role": "system", "content": system_prompt_doctors_llm}, {"role": "user", "content": message_text}]
    response = openai_client.chat.completions.create(model=MODEL_NAME_LLM, messages=messages, tools=tools_doctors)
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_response = handle_doctors_call(message)
        messages.append(message)
        messages.append(tool_response)
        final_response = openai_client.chat.completions.create(model=MODEL_NAME_LLM, messages=messages)
        return final_response.choices[0].message.content
    return ""  # Return empty string if no tool was called


# --- Stage 3: Final Report Generation with RAG ---
SYSTEM_TEMPLATE = """
You are a highly professional Persian-speaking medical assistant.
You are given:
1. A disease prediction output.
2. A list of top 5 doctors.
3. Medical knowledge (RAG context).
Your task is to generate a complete, well-explained, and clear **medical report in Persian (Farsi)**. The report **must** include all of the following:
---
🔹 **1. Disease Name and Probability Explanation:**
- Clearly state the disease name and its probability in Persian.
---
🔹 **2. Scientifically Accurate Description of the Disease:**
Using the RAG context, explain the disease, symptoms, risks, and risk groups.
---
🔹 **3. Detailed, Evidence-Based Medical Advice:**
- Based on the RAG context, provide practical and actionable recommendations.
- **For each recommendation, you MUST provide a brief but clear explanation of WHY it is important and HOW it helps the patient.**
- For example, if you recommend the 'DASH diet', you must also explain what it consists of (e.g., a diet rich in fruits, vegetables, whole grains, and low in salt and saturated fat).
- Structure your advice under the following subheadings:
  - **Medical Actions:** (e.g., Follow up with a specialist, perform regular check-ups).
  - **Nutrition:** (e.g., Recommended diets, foods to eat or avoid).
  - **Lifestyle Changes:** (e.g., Exercise, smoking cessation, stress management).

⚠️ **Crucially, all explanations and advice must be strictly based on the provided RAG context. Do not invent any information.**
---
    🔹 4. List of Top 5 Doctors in the Patient’s Region:
    
    - Directly include the pre-formatted list of doctors as provided in the input. Do **not** rephrase, translate, summarize, or omit any field.
    
    For each doctor, display the following details exactly as given:
    
    - 👨‍⚕️ Doctor’s Name (in Persian, from `name`)
    - 🏥 Medical ID (from `medical_id`)
    - 🩺 Specialty (from `specialty`)
    - 📍 Locations (from `locations` — include all available addresses)
    - 🛎️ Services Offered (You should write the services provided in a readable paragraph without summarizing them from `services`, as-is)
    - 💯 Score Summary (from `score_summary`)
    - 💳 Accepted Insurances (from `accepted_insurances`)
    - 🔗 Reservation Link: You can hide the **exact URL** in a word like لینک نوبت دهی.  
      Example:  
      Reservation Link: https://doctoreto.com/doctor/dr-seyed-mokhtar-javdan-nezhad/MRonzZ

    ---
    🛑 Do not include tool names, raw data, or JSON. Write everything fluently and professionally in Persian.
    🛑 To improve readability, you **MUST** insert a horizontal separator line (`---`) after completing each of the 4 main sections of the report.
    ---
    🔹 **5. Recommended Drugs and Supplements:**

    - Directly include the pre-fetched product list exactly as provided.
    - Do **not** translate, rephrase, or summarize any field.
    - Present each product with the following details:
    
      - 💊 Product Name (from `title` + 'name')
      - 💰 Price (from `price`)
      - ℹ️ Product Description (from `info`)
      Summarize this section.
      - 🔗 Direct Purchase Link: Use a phrase like “product link” and embed the actual URL behind it.  
        Example: Product Link: https://mokamelkhoone.com/products/now-foods-omega-3/
    
    🛑 Do not invent or assume any drug or supplement recommendations.
    🛑 Do not translate, modify, or shorten any field. Keep the format consistent and clear.
    ---
    🔹 **6. داروخانه‌های در دسترس (Available Drugstores):**

    - Provide information about the most relevant nearby drugstore(s) based on the user's location and the availability of the recommended drug(s).
    - If a specific drug is available, include all details of the matching drugstore.
    - If the drug is not available, show the nearest available drugstore with a note that the medication might not be in stock.
    
    For each drugstore, include:
    
    - 🏪 Drugstore Name
    - 🗺️ Province
    - 📍 City
    - 🏠 Address
    - ☎️ Phone Number
    - 🗓️ Available Days
    - 💊 Drug Generic Name

    
    🛑 Do not guess missing information.  
    🛑 Do not translate or reformat fields.  
    ✅ Present everything in Persian, clearly and politely.
        ---
    📥 **Input Data**:
    🔸 **Input from Tools (Disease & Doctors):**
    {question}
    🔸 **Medical Knowledge (RAG Context):**
    {context}
    ---
    📄 **Medical Report (in Persian):**
    """
QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=SYSTEM_TEMPLATE)


def chat_with_rag_llm(tools_response, doctors_response, drugs_response):
    if not vectorstore:
        return "خطا: پایگاه دانش به درستی بارگذاری نشده است. امکان تولید گزارش کامل وجود ندارد."

    llm = ChatOpenAI(base_url=BASE_URL, api_key=METIS_API_KEY, temperature=0.8, model_name=MODEL_NAME_LLM)
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


def slugify(text):
    return re.sub(r'[^a-zA-Z0-9]+', '-', text.lower()).strip('-')


def convert_persian_to_english(text):
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    english_digits = '0123456789'
    return text.translate(str.maketrans(persian_digits, english_digits))


def scrape_drugs(medicine_name):
    slug = slugify(medicine_name)
    name = slug
    url = f"https://mokamelkhoone.com/products/now-foods-{slug}/"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract price
        price_tag = soup.select_one("p.price bdi")
        if not price_tag:
            raise Exception("Price element not found")

        raw_price = price_tag.text.strip()
        price_only = re.sub(r"[^\d۰-۹٬,]", "", raw_price)
        price = convert_persian_to_english(price_only)

        # Extract product info
        info_tag = soup.select_one(".product-content-c")
        if not info_tag:
            raise Exception("Product info not found")
        info_text = info_tag.text.strip()

        # Extract title
        title_tag = soup.select_one("h1.elementor-heading-title")
        product_title = title_tag.text.strip() if title_tag else ""
        final_response = {
            "name": name,
            "url": url,
            "price": price,
            "info": info_text,
            "title": product_title
        }

        return json.dumps(final_response)

    except Exception as e:
        return json.dumps({
            "url": url,
            "error": str(e)
        })


scrape_drugs_function = {
    "name": "scrape_drugs",
    "description": (
        "Fetch product details for a given supplement or drug name from mokamelkhoone.com. "
        "Returns the title, current price, product info, and a direct URL. "
        "Use this when a user wants to look up the price or details of a supplement product."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "medicine_name": {
                "type": "string",
                "description": (
                    "Name of the medicine or supplement (in English). Example:\n"
                    "- Magnesium citrate"

                )
            }
        },
        "required": ["medicine_name"],
        "additionalProperties": False
    }
}
tools_drugs = [{"type": "function", "function": scrape_drugs_function}]
system_prompt_drugs_llm = """
You are a smart medicine assistant.

Your task:

1. If the user's message or tool output contains a diagnosis or risk statement like:
   "⚠️ Based on the model, the patient has a 74.5% probability of having hypertension."
   – then detect the name of the condition (e.g., "hypertension").

2. If no such condition or risk is detected (i.e., no significant probability or diagnosis is mentioned), then:
   ❌ Do not call any tools.
   ❌ Do not return any doctor information.
   ✅ Simply do nothing.

3. If a condition is detected, proceed as follows:

   - Hypertension → Offer Magnesium citrate as a medicine


4. ✅ Return only the raw JSON output from the scrape_drugs tool.
   ❌ Do not include any explanations, summaries, or extra text.
"""
# --- CSV Vectorstore ---
print("start")
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
print(f"[CSV] Vectors: {csv_vectorstore.index.ntotal}")

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage

import json


def chat_with_drugs_llm(user_message):
    csv_retriever = csv_vectorstore.as_retriever()

    # 2. Setup LLM
    llm = ChatOpenAI(
        base_url="https://api.metisai.ir/openai/v1",
        api_key=METIS_API_KEY,
        temperature=0.2,
        model_name=MODEL_NAME_LLM
    )

    # 3. System prompt to guide behavior
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

    # 4. Memory for dialogue context
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 5. Tool: scrape_drugs
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

    # 6. Tool: RAG search in CSV (returns drugstore info)
    def rag_tool_func(drug_name: str) -> str:
        # اول بررسی می‌کنیم که آیا دارو وجود داره یا نه
        query_primary = f"Find a drugstore near to the region of patient that has '{drug_name}' in stock. For example, if the region was vanak you should search in vanak square at address section in CSV."
        result = RetrievalQA.from_chain_type(llm=llm, retriever=csv_retriever).run(query_primary)

        # سپس بدون توجه به اینکه دارو پیدا شده یا نه، حتماً نزدیک‌ترین داروخانه‌ها رو هم می‌گیریم
        fallback_query = "List the nearest drugstores located in patient's region regardless of drug availability. Return their address and contact details so the patient can visit them in person to ask about the drug. Pay attention to address in CSV not to province or city."
        fallback_result = RetrievalQA.from_chain_type(llm=llm, retriever=csv_retriever).run(fallback_query)

        # اگر دارو پیدا شد
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

    # 7. Initialize LangChain agent with tools
    agent = initialize_agent(
        tools=[scrape_tool, rag_tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        agent_kwargs={"system_message": system_prompt},
        verbose=True
    )

    # 8. Run agent
    return agent.run(user_message)


# ==============================================================================
# 🏁 Section 4: Main Pipeline Execution Function
# ==============================================================================

def run_health_analysis_pipeline(profile_text_summary):
    """
    The main entry point for the AI pipeline.
    It orchestrates the three stages: predict, find doctors, and generate report.
    """
    print("\n--- Starting AI Pipeline for new request ---")

    # Stage 1: Predict hypertension risk
    print("Stage 1: Calling Tools LLM...")
    tools_response_str = chat_with_tools_llm(profile_text_summary)
    try:
        tools_response_json = json.loads(tools_response_str)
        if not tools_response_json:  # Handle case where LLM returns empty {}
            raise json.JSONDecodeError("Empty JSON object returned", tools_response_str, 0)
        print("Stage 1 Success:", tools_response_json.get('output'))
    except json.JSONDecodeError:
        print(f"--- Raw Response from Stage 1 LLM ---\n{tools_response_str}\n------------------------------------")
        print("Stage 1 Failed: No valid JSON returned. Aborting.")
        return "گزارش به دلیل عدم تشخیص وضعیت پزشکی توسط مدل اول، تولید نشد."

    # Stage 2: Find doctors based on the prediction
    print("\nStage 2: Calling Doctor Finder LLM...")
    # Pass the JSON string from stage 1 to stage 2
    doctors_response_str = chat_with_doctors_llm(tools_response_str)

    # To prevent errors, ensure we have a valid JSON string (list)
    if not doctors_response_str or not doctors_response_str.strip().startswith('['):
        print("Stage 2 Note: No doctors found or tool not called. Using default empty list for final report.")
        doctors_response_str = "[]"
    else:
        print("Stage 2 Success: Doctor list retrieved.")

    print("\nStage 3: Calling Drug And Drugstore Finder LLM...")
    drugs_response = chat_with_drugs_llm(tools_response_str)
    print("Stage 3 Success: Drug And Drugstore list retrieved.")

    # Stage 3: Generate the final comprehensive report using RAG
    print("\nStage 4: Calling Final Report Generation LLM with RAG...")
    final_report = chat_with_rag_llm(tools_response_json, doctors_response_str, drugs_response)
    print("Stage 4 Success: Final report generated.")

    print("--- AI Pipeline Finished ---")
    return final_report
