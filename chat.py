import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_DIR = "chroma_db"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MAX_CONTEXT_CHARS = 12000
MAX_OUTPUT_TOKENS = 512
PRO_VC_URL = "https://uetmardan.edu.pk/uetm/Site/provcmessage"
VC_URL = "https://uetmardan.edu.pk/uetm/Site/vcmessage"
ADMISSION_PROCESS_URL = (
    "https://www.uetmardan.edu.pk/uetm/Admissions/applicationprocess"
)

FEE_TERMS = (
    "admission fee",
    "admission fees",
    "application fee",
    "application fees",
    "fee structure",
    "tuition fee",
    "undergraduate fee",
    "prospectus fee",
    "bs computer science fee",
    "fee for bs computer science",
    "how much does admission cost",
)

ENTRANCE_TEST_TERMS = (
    "entrance test",
    "entry test",
    "etea",
    "engineering admission test",
    "engineering admissions test",
)

BS_COMPUTER_SCIENCE_FEE_RESPONSE = """
### BS Computer Science fee

The documented fee for BS Computer Science is **Rs. 2,000**. This is the
undergraduate application processing and prospectus fee, not the complete
tuition or semester fee structure.

For current tuition and other charges, check the official [UET Mardan
admission process page](https://www.uetmardan.edu.pk/uetm/Admissions/applicationprocess).
""".strip()

ENTRANCE_TEST_RESPONSE = (
    "Engineering applicants must take the ETEA entrance test, "
    "conducted by the Educational Testing and Evaluation Agency of the "
    "Government of Khyber Pakhtunkhwa."
)

MULTI_APPLICATION_TERMS = (
    "more than one program",
    "multiple programs",
    "one program",
    "another program",
    "different program",
    "programs can i apply",
)

DEPARTMENT_LIST_TERMS = (
    "what departments",
    "which departments",
    "list of departments",
    "departments currently",
    "operating departments",
    "departments at uet",
)

CURRENT_PROSPECTUS_FILENAME = (
    "uetm_assets_prospectous_undergraduate_Prospectus-2026-27.pdf_pdf.txt"
)

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Make sure your .env file has GOOGLE_API_KEY=your_key")

# Load the same embedding model used to build the index
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the existing ChromaDB
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# Set up the Gemini client. Chat streaming avoids the deprecated direct
# generate_content path used by the LangChain wrapper.
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions about UET Mardan
(University of Engineering & Technology, Mardan) using ONLY the context provided below.

Rules:
- If the answer is in the context, answer clearly and directly.
- Always express information in your own words. Do not copy or closely
    paraphrase full sentences from the context verbatim — restate facts
    naturally, even when using the same key terms.
- For questions about applying to more than one program, distinguish
    between multiple academic programs and multiple admission/quota categories.
    The prospectus explicitly requires separate applications for each additional
    eligible quota category, but do not claim that applicants can submit one
    application for multiple programs unless the context says so.
- For department-list questions, use department pages and the university home
    page as the authoritative list. Prefer those named department entries over
    historical messages about departments being planned or launched.
- If the context does not contain the answer, say you don't have that information
  and suggest the user check the official website uetmardan.edu.pk.
- Do not make up facts that are not in the context.
- Keep answers concise and well-organized.

Context:
{context}

Question: {question}

Answer:
""")


def call_gemini(prompt, retry=True):
    chat = gemini_client.chats.create(
        model=GEMINI_MODEL,
        config={
            "temperature": 0.2 if retry else 0.4,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )

    text_parts = []
    finish_reason = None
    for chunk in chat.send_message_stream(message=prompt):
        text_parts.append(chunk.text or "")
        if chunk.candidates:
            reason = chunk.candidates[0].finish_reason
            if reason:
                finish_reason = str(reason)
                print(f"[DEBUG] finish_reason: {reason}")

    result = "".join(text_parts).strip()

    if (
        retry
        and finish_reason
        and "MAX_TOKENS" not in finish_reason
        and finish_reason not in ("STOP", "1", "FinishReason.STOP")
    ):
        nudge = (
            prompt
            + "\n\nIMPORTANT: Paraphrase everything in your own words. "
              "Do not reuse phrasing from the context."
        )
        return call_gemini(nudge, retry=False)

    return result


def answer_question(question, k=10):
    """Retrieve the top-k relevant chunks and generate an answer."""
    normalized_question = question.lower().replace("-", " ")
    if any(term in normalized_question for term in FEE_TERMS):
        return BS_COMPUTER_SCIENCE_FEE_RESPONSE, [ADMISSION_PROCESS_URL]
    elif any(term in normalized_question for term in ENTRANCE_TEST_TERMS):
        results = vectordb.similarity_search(
            "entrance test engineering programs ETEA Educational Testing Evaluation Agency",
            k=4,
            filter={"filename": CURRENT_PROSPECTUS_FILENAME},
        )
    elif any(term in normalized_question for term in MULTI_APPLICATION_TERMS):
        results = []
        seen = set()
        for query in (
            "candidates applying for more than one category separate applications",
            "separate application form additional category admission",
            question,
        ):
            for doc in vectordb.similarity_search(query, k=4):
                document_key = (doc.metadata.get("source_url"), doc.page_content)
                if document_key not in seen:
                    seen.add(document_key)
                    results.append(doc)
        results = results[:10]
    elif any(term in normalized_question for term in DEPARTMENT_LIST_TERMS):
        results = []
        seen = set()
        targeted_queries = (
            ("departments contents UET Mardan 2026 2027", {"filename": CURRENT_PROSPECTUS_FILENAME}),
            ("departments", {"filename": "home.txt"}),
            ("Department Of Electrical Engineering academic programs", None),
            ("Department Of Computer Science academic programs", None),
            ("Department Of Mechanical Engineering academic programs", None),
            ("Department Of Civil Engineering academic programs", None),
            ("Department Of Telecommunication Engineering academic programs", None),
        )
        for query, filter_metadata in targeted_queries:
            search_kwargs = {"k": 3}
            if filter_metadata:
                search_kwargs["filter"] = filter_metadata
            for doc in vectordb.similarity_search(query, **search_kwargs):
                document_key = (doc.metadata.get("source_url"), doc.page_content)
                if document_key not in seen:
                    seen.add(document_key)
                    results.append(doc)
        results = results[:12]
    elif "pro vice chancellor" in normalized_question:
        exact_results = vectordb.similarity_search(
            "Pro-Vice Chancellor", k=3, filter={"source_url": PRO_VC_URL}
        )
        general_results = vectordb.similarity_search(question, k=min(k, 6))
        results = exact_results + [
            doc for doc in general_results
            if doc.metadata.get("source_url") != PRO_VC_URL
        ]
    elif "vice chancellor" in normalized_question or " vc " in f" {normalized_question} ":
        exact_results = vectordb.similarity_search(
            "Vice Chancellor", k=3, filter={"source_url": VC_URL}
        )
        general_results = vectordb.similarity_search(question, k=min(k, 6))
        results = exact_results + [
            doc for doc in general_results
            if doc.metadata.get("source_url") != VC_URL
        ]
    else:
        results = vectordb.similarity_search(question, k=min(k, 6))

    if any(term in normalized_question for term in ENTRANCE_TEST_TERMS):
        sources = list({doc.metadata.get("source_url", "Unknown") for doc in results})
        return ENTRANCE_TEST_RESPONSE, sources

    if not results:
        return "I couldn't find relevant information for that question.", []

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])[:MAX_CONTEXT_CHARS]
    sources = list({doc.metadata.get("source_url", "Unknown") for doc in results})

    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
    try:
        content = call_gemini(prompt)
    except errors.ClientError as error:
        error_text = str(error).lower()
        if error.code == 429 or "quota" in error_text or "resource_exhausted" in error_text:
            content = (
                "Gemini is temporarily unavailable because the API quota has been "
                "exhausted. Relevant information from the local UET Mardan "
                "knowledge base:\n\n"
                + context_text[:2500]
                + "\n\nPlease try again later or check the official UET Mardan website."
            )
        else:
            content = (
                "Gemini is temporarily unavailable. Relevant information from the "
                "local UET Mardan knowledge base:\n\n"
                + context_text[:2500]
            )

    return content, sources


if __name__ == "__main__":
    print("UET Mardan Chatbot (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer, sources = answer_question(question)
        print(f"\nBot: {answer}\n")
        if sources:
            print("Sources:")
            for s in sources:
                print(f"  - {s}")
        print()