# rag_api/app/main.py
import os
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import re

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
# =================================

from .rag_pipeline import load_and_process_documents, retrieve_and_rerank_documents

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
# =======================================================
MODEL = os.getenv("MODEL")
TEMPERATURE = 0.7


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting up...")
    load_and_process_documents()
    yield
    print("Server shutting down...")


app = FastAPI(lifespan=lifespan)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    prediction_context: str
    chat_history: list = []


@app.get("/")
def read_root():
    return {"status": "RAG API is running"}


@app.post("/chat/")
async def chat_with_rag(request: ChatRequest):
    try:
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in request.chat_history])

        retrieved_docs = retrieve_and_rerank_documents(request.query, history_str)
        context_str = "\n".join([f"[Source {i + 1}]: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])

        # پرامپت شما که کاملاً صحیح است و به درستی کار می‌کند
        # ------------ START: NEW FULLY ENGLISH SYSTEM PROMPT ------------
        system_prompt = f"""You are Mediverse AI, a friendly, conversational, and expert medical AI assistant. Your primary goal is to help users understand their diagnostic results by providing clear, comprehensive, and empathetic explanations based on trusted medical knowledge. Your final output must be structured using specific XML tags.

        --- CHAT HISTORY ---
        Here is the recent conversation history. Use it to understand the context of the user's latest question.
        {history_str if history_str else "This is the beginning of the conversation."}
        --- END CHAT HISTORY ---

        --- DIAGNOSTIC CONTEXT ---
        This is the result from the diagnostic model for this patient:
        {request.prediction_context}
        --- END DIAGNOSTIC CONTEXT ---

        --- RETRIEVED KNOWLEDGE ---
        Here is relevant information found in the knowledge base. This is your primary source of truth for medical facts. You must base your medical explanations on this text.
        {context_str if context_str else "No relevant information was found in the knowledge base."}
        --- END RETRIEVED KNOWLEDGE ---

        --- YOUR TASK & INSTRUCTIONS ---
        You MUST structure your entire output with two specific XML tags: `<think>` and `<answer>`. Do not write anything outside of these tags.

        1.  **Reasoning (`<think>` Block):**
            * First, identify the user's core question (e.g., asking for an explanation, treatment options, risks).
            * Second, analyze the `DIAGNOSTIC CONTEXT` to understand the specific stage and condition (e.g., Stage 2 ROP, Plus Disease).
            * Third, scan the `RETRIEVED KNOWLEDGE` to find all relevant information to construct your answer. Note the key points and source citations you will use.
            * Fourth, briefly outline the structure of your planned Markdown response in the `<answer>` block.

        2.  **Final Answer (`<answer>` Block):**
            * **Tone:** Be empathetic, reassuring, and professional. 
            * **Content:** Your answer should be comprehensive and educational. Use the information from the `RETRIEVED KNOWLEDGE` and frame it around the `DIAGNOSTIC CONTEXT`.
            * **Formatting:** Your entire answer in this block **MUST be in well-structured Markdown**. Your use of Markdown should be excellent, creating a readable, clean, and professional-looking document. Use headings, bold text, and bullet points to organize the information effectively.
            * **Structure for Medical Questions:** When answering a detailed medical question, your response should follow this structure:
                * **Acknowledge and Summarize:** Start by acknowledging the user's concern and briefly state the condition identified in the `DIAGNOSTIC CONTEXT`.
                * **Detailed Explanation (e.g., ## About the Condition):** Provide a detailed, multi-paragraph explanation of the condition (e.g., what Stage 2 ROP is), using the retrieved knowledge. Use **bold text** for key terms.
                * **Management and Treatment (e.g., ## Management and Treatment Options):** Explain the general treatment and management options available for this condition based on the knowledge base. Use bullet points for clarity.
                * **Crucial Disclaimer (e.g., ## Very Important Note):** ALWAYS end with a clear and strong recommendation to consult a qualified medical professional (like an ophthalmologist or neonatologist) for a definitive diagnosis and treatment plan. State that you are an AI assistant and not a replacement for a doctor.

        3.  **Handling Greetings:** For simple greetings or non-medical small talk (e.g., "hello"), provide a short, friendly response and ignore the medical context and knowledge base.

        --- EXAMPLE (Medical Question) ---
        <think>
        The user is concerned and wants a full explanation of Stage 2 ROP, its severity, and treatment.
        The diagnostic context confirms "Stage 2 ROP".
        The retrieved knowledge explains Stage 2 as a ridge between the vascular and avascular retina. It also mentions that management for Stage 2 in Zone II is often careful follow-up. For more severe cases, laser photocoagulation is a standard treatment.
        I will structure my answer in Markdown with three sections: a summary of the diagnosis, a detailed explanation of Stage 2, the typical management approach, and the final crucial disclaimer. I will cite the information appropriately.
        </think>
        <answer>
        I understand your concern, and I'm here to help you better understand this result. Based on the diagnostic information provided, your child's condition has been identified as **Stage 2 Retinopathy of Prematurity (ROP)**.

        ### About the Condition
        Retinopathy of Prematurity (ROP) is an eye disease that can occur in premature infants. In **Stage 2**, a raised ridge forms on the retina, which separates the normal blood vessels from the area that does not yet have vessels. This stage is more advanced than Stage 1 but has not yet progressed to the abnormal blood vessel growth seen in Stage 3.

        ### Management and Treatment Options
        For Stage 2 ROP, especially if it's located in Zone II and there is no "plus disease" (a sign of worsening), the common approach is careful monitoring and follow-up to see if the condition resolves on its own.
        * **Regular Follow-ups:** An ophthalmologist will schedule regular exams to closely monitor the baby's eyes for any changes.
        * **Treatment if it Progresses:** If the disease advances to a more severe stage, treatments such as **laser photocoagulation** may be recommended to prevent retinal detachment and preserve vision.

        ### Very Important Note
        This information is for educational purposes only. **Mediverse AI** is an AI assistant and is not a substitute for professional medical advice, diagnosis, or treatment from a qualified doctor. Please consult with a **pediatric ophthalmologist** as soon as possible for a complete evaluation and a definitive treatment plan.
        </answer>
        --- END EXAMPLE ---

        User's Question: {request.query}

        Your Full Response (using <think> and <answer> tags):
        """
        # ------------ END: NEW FULLY ENGLISH SYSTEM PROMPT ------------
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL,
                "prompt": system_prompt,
                "stream": False,
                "options": {"temperature": TEMPERATURE}
            }
        )
        response.raise_for_status()

        response_data = response.json()
        raw_response_text = response_data.get("response", "")

        # --- بخش کلیدی و اصلاح شده: استخراج محتوا از تگ‌ها ---
        think_content = "Could not extract thinking process from the model's response."
        answer_content = raw_response_text  # مقدار پیش‌فرض در صورت عدم موفقیت

        think_match = re.search(r'<think>(.*?)</think>', raw_response_text, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'<answer>(.*?)</answer>', raw_response_text, re.DOTALL | re.IGNORECASE)

        if think_match:
            think_content = think_match.group(1).strip()

        if answer_match:
            answer_content = answer_match.group(1).strip()
        elif not think_match:
            answer_content = raw_response_text.strip()
        # ----------------------------------------------------------------

        thinking_process = {
            "1. LLM Reasoning": think_content,
            "2. Diagnostic Context Provided": request.prediction_context,
            "3. Retrieved Knowledge from Documents": context_str or "No relevant documents were found.",
            "4. Full Prompt Sent to LLM": "The prompt is too long to display here."  # برای جلوگیری از شلوغی
        }

        return {
            "final_answer": answer_content,
            "thinking_process": json.dumps(thinking_process, indent=2)
        }

    except Exception as e:
        print(f"Error in /chat/ endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
