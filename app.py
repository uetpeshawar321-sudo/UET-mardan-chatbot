import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UET Mardan AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(30, 136, 229, 0.08),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(76, 175, 80, 0.06),
                transparent 30%
            );
    }

    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 6rem;
    }


    /* ---------- Header ---------- */

    .main-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 8px;
    }

    .header-logo {
        width: 58px;
        height: 58px;
        min-width: 58px;
        border-radius: 18px;

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 30px;

        background: linear-gradient(
            135deg,
            #1769aa,
            #2e8b57
        );

        box-shadow:
            0 10px 25px rgba(23, 105, 170, 0.20);
    }

    .header-title {
        font-size: 30px;
        font-weight: 750;
        line-height: 1.1;
        margin: 0;
    }

    .header-subtitle {
        font-size: 14px;
        opacity: 0.70;
        margin-top: 5px;
    }


    /* ---------- Welcome Card ---------- */

    .welcome-card {
        margin: 28px 0 22px 0;
        padding: 30px;

        border-radius: 24px;

        background:
            linear-gradient(
                135deg,
                rgba(23, 105, 170, 0.10),
                rgba(46, 139, 87, 0.07)
            );

        border: 1px solid rgba(120, 120, 120, 0.15);

        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.04);
    }

    .welcome-title {
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .welcome-text {
        opacity: 0.75;
        font-size: 15px;
        line-height: 1.7;
    }


    /* ---------- Suggestion Cards ---------- */

    .suggestion-title {
        font-size: 14px;
        font-weight: 650;
        margin-bottom: 10px;
        opacity: 0.75;
    }

    .suggestion-card {
        padding: 15px 17px;
        border-radius: 16px;

        border: 1px solid rgba(120, 120, 120, 0.14);

        background: rgba(128, 128, 128, 0.05);

        min-height: 82px;

        transition: transform 0.2s ease,
                    box-shadow 0.2s ease;
    }

    .suggestion-card:hover {
        transform: translateY(-2px);
        box-shadow:
            0 8px 20px rgba(0, 0, 0, 0.07);
    }

    .suggestion-icon {
        font-size: 22px;
        margin-bottom: 5px;
    }

    .suggestion-text {
        font-size: 13px;
        line-height: 1.4;
    }


    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(120, 120, 120, 0.12);
    }

    .sidebar-logo {
        text-align: center;
        font-size: 48px;
        margin-top: 8px;
        margin-bottom: 5px;
    }

    .sidebar-title {
        text-align: center;
        font-size: 19px;
        font-weight: 700;
    }

    .sidebar-subtitle {
        text-align: center;
        font-size: 12px;
        opacity: 0.65;
        margin-bottom: 25px;
    }

    .sidebar-section {
        font-size: 13px;
        font-weight: 700;
        opacity: 0.75;
        margin-top: 20px;
        margin-bottom: 8px;
    }

    .info-card {
        padding: 13px;
        border-radius: 14px;

        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.12);

        margin-bottom: 8px;
    }

    .info-label {
        font-size: 11px;
        opacity: 0.6;
        margin-bottom: 3px;
    }

    .info-value {
        font-size: 13px;
        font-weight: 600;
    }


    /* ---------- Chat Messages ---------- */

    [data-testid="stChatMessage"] {
        padding: 8px 0;
    }

    [data-testid="stChatMessageContent"] {
        border-radius: 18px;
        padding: 12px 16px;
    }


    /* ---------- Chat Input ---------- */

    [data-testid="stChatInput"] {
        padding-bottom: 10px;
    }


    /* ---------- Typing Indicator ---------- */

    .typing-indicator {
        display: inline-flex;
        align-items: center;
        gap: 5px;

        padding: 10px 15px;

        border-radius: 18px;

        background: rgba(128, 128, 128, 0.08);

        font-size: 13px;
        opacity: 0.75;
    }

    .typing-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;

        background: currentColor;

        animation: typing 1.2s infinite ease-in-out;
    }

    .typing-dot:nth-child(2) {
        animation-delay: 0.15s;
    }

    .typing-dot:nth-child(3) {
        animation-delay: 0.30s;
    }

    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.4;
        }

        30% {
            transform: translateY(-4px);
            opacity: 1;
        }
    }


    /* ---------- Mobile ---------- */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .header-title {
            font-size: 23px;
        }

        .header-logo {
            width: 48px;
            height: 48px;
            min-width: 48px;
            font-size: 25px;
            border-radius: 15px;
        }

        .welcome-card {
            padding: 22px;
            border-radius: 20px;
        }

        .welcome-title {
            font-size: 21px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

CHROMA_DIR = "chroma_db"

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)

MAX_CONTEXT_CHARS = 12000
MAX_OUTPUT_TOKENS = 512

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

VC_URL = "https://uetmardan.edu.pk/uetm/Site/vcmessage"

PRO_VC_URL = "https://uetmardan.edu.pk/uetm/Site/provcmessage"

TRANSPORT_URL = (
    "https://www.uetmardan.edu.pk/uetm/assets/files/downloads/"
    "11122020_UETM_Transportation_Form.pdf"
)

ADMISSION_PROCESS_URL = (
    "https://www.uetmardan.edu.pk/uetm/Admissions/applicationprocess"
)
ADMISSION_PROCESS_FILENAME = "uetm_Admissions_applicationprocess.txt"

ADMISSION_PROCESS_TERMS = (
    "admission process",
    "admission procedure",
    "application procedure",
    "how to apply",
    "apply for admission",
    "get admission",
    "admission application",
    "apply for uet",
    "application process",
)

ABOUT_UET_TERMS = (
    "what is uet mardan",
    "what's uet mardan",
    "tell me about uet mardan",
    "about uet mardan",
    "uet mardan overview",
    "information about uet mardan",
)

UNDERGRADUATE_PROGRAM_TERMS = (
    "bachelor degree programs",
    "bachelor programs",
    "undergraduate programs",
    "bs programs",
    "bsc programs",
    "what programs does uet",
    "which programs does uet",
    "programs offered",
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

FEE_TERMS = (
    "admission fee",
    "admission fees",
    "application fee",
    "application fees",
    "fee structure",
    "tuition fee",
    "undergraduate fee",
    "prospectus fee",
    "how much does admission cost",
)

ENTRANCE_TEST_TERMS = (
    "entrance test",
    "entry test",
    "etea",
    "engineering admission test",
    "engineering admissions test",
)

FEE_RESPONSE = """
### Undergraduate admission application fees

- **BS Computer Science:** Rs. 2,000
- **Engineering programs:** Rs. 1,500

These are the application processing and prospectus fees, not total tuition
fees. The payment instructions are provided on the official [admission process
page](https://www.uetmardan.edu.pk/uetm/Admissions/applicationprocess).
"""

ENTRANCE_TEST_RESPONSE = (
    "Engineering applicants must take the ETEA entrance test, "
    "conducted by the Educational Testing and Evaluation Agency of the "
    "Government of Khyber Pakhtunkhwa."
)

ADMISSION_PROCESS_RESPONSE = """
### UET Mardan undergraduate admission process

#### Non-engineering programs (BS Computer Science)

1. Complete the online application form at [ugadmissions](https://www.uetmardan.edu.pk/ugadmissions).
2. Upload clear scanned copies of the required documents while completing the form.
3. Deposit **Rs. 2,000** as the application processing and prospectus fee at any Bank of Khyber (BOK) branch in Khyber-Pakhtunkhwa:
    - **Account:** `PK55KHYB 0179003004139436`
    - **Branch code:** `0179`
4. Erstwhile FATA candidates may also apply under the Open and Rationalized schemes. For FATA reserved-quota seats, contact the Directorate of Admissions at UET Mardan or UET Peshawar for an application form.

#### Engineering programs

1. Complete the online application form at [engineering admissions](https://www.uetmardan.edu.pk/engineering).
2. Upload clear scanned copies of the required documents while completing the form.
3. Deposit **Rs. 1,500** as the application processing and prospectus fee at any Bank of Khyber (BOK) branch in Khyber-Pakhtunkhwa:
    - **Account:** `PK55KHYB 0179003004139436`
    - **Branch code:** `0179`
4. Erstwhile FATA candidates may also apply under the Open and Rationalized schemes of BSc Engineering. For FATA reserved-quota seats, contact the Directorate of Admissions at UET Mardan or UET Peshawar for an application form.
"""

ABOUT_UET_RESPONSE = """
### About UET Mardan

The University of Engineering & Technology Mardan (UET Mardan) is an emerging public-sector university recognized by the Pakistan Engineering Council (PEC) and the Higher Education Commission (HEC).

The university offers programs through departments including:

- Electrical Engineering
- Computer Software Engineering
- Telecommunication Engineering
- Computer Science
- Mechanical Engineering
- Civil Engineering
- Natural Sciences and Humanities

UET Mardan also has a Center of Artificial Intelligence.
"""

UNDERGRADUATE_PROGRAM_RESPONSE = """
### UET Mardan bachelor's degree programs

According to the UET Mardan undergraduate prospectus and admission schedule, the university offers:

#### Engineering programs

- BSc Electrical Engineering
- BSc Civil Engineering
- BSc Mechanical Engineering
- BSc Telecommunication Engineering
- BSc Computer Software Engineering

#### Computing and advanced technology programs

- BS Computer Science
- BS Computer Science with specialization in Cyber Security
- BS Computer Science with specialization in Data Science
- BS Computer Science with specialization in Bioinformatics
- BS Artificial Intelligence

Program availability can change by admission session. Check the current [UET Mardan admission notices](https://uetmardan.edu.pk/) before applying.
"""

VICE_CHANCELLOR_RESPONSE = "Prof. Dr. Gul Muhammad Khan is the current Vice Chancellor of UET Mardan."

HOSTEL_RESPONSE = """
Yes. UET Mardan provides hostel accommodation for outstation students.

- Three hostels are currently available: two for male students and one for female students.
- They have capacity for approximately 380 students.
- Hostel accommodation is limited and allocated according to merit and availability, so it cannot be guaranteed for every student.
"""

KNOWN_FOR_RESPONSE = """
UET Mardan is known for engineering and technology education, with a growing focus on:

- Artificial intelligence and intelligent technologies
- Applied research and innovation
- Smart energy, IoT, transportation, and sustainable digital solutions
- Entrepreneurship and collaboration with industry
"""


def local_quota_fallback(question, context):
    normalized_question = question.lower()

    if any(term in normalized_question for term in UNDERGRADUATE_PROGRAM_TERMS):
        return UNDERGRADUATE_PROGRAM_RESPONSE.strip()

    if any(term in normalized_question for term in ENTRANCE_TEST_TERMS):
        return ENTRANCE_TEST_RESPONSE

    if any(term in normalized_question for term in ABOUT_UET_TERMS):
        return ABOUT_UET_RESPONSE.strip()

    if "vice chancellor" in normalized_question or " vc " in f" {normalized_question} ":
        return VICE_CHANCELLOR_RESPONSE

    if "hostel" in normalized_question or "accommodation" in normalized_question:
        return HOSTEL_RESPONSE.strip()

    if "known for" in normalized_question or "known" in normalized_question:
        return KNOWN_FOR_RESPONSE.strip()

    if context:
        return (
            "Gemini is temporarily unavailable, but the relevant information "
            "from the UET Mardan knowledge base is:\n\n"
            + context[:2500]
        )

    return (
        "Gemini is temporarily unavailable and I could not find a local answer. "
        "Please try again later or check the official UET Mardan website."
    )

CURRENT_PROSPECTUS_FILENAME = (
    "uetm_assets_prospectous_undergraduate_Prospectus-2026-27.pdf_pdf.txt"
)


# ============================================================
# API KEY CHECK
# ============================================================

if not GOOGLE_API_KEY:
    st.error(
        "🔑 GOOGLE_API_KEY is missing. "
        "Please add it to your .env file and restart the application."
    )
    st.stop()


# ============================================================
# LOAD RAG SYSTEM
# ============================================================

@st.cache_resource
def load_rag():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    gemini_client = genai.Client(
        api_key=GOOGLE_API_KEY
    )

    return vectordb, gemini_client


try:
    vectordb, gemini_client = load_rag()

except Exception as error:
    st.error(
        f"⚠️ Could not initialize the chatbot: {error}"
    )
    st.stop()


# ============================================================
# PROMPT
# ============================================================

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant for UET Mardan.

Answer questions about UET Mardan using ONLY the context provided below.

Rules:
- If the answer is in the context, answer clearly and directly.
- Always express information in your own words. Do not copy or closely
    paraphrase full sentences from the context verbatim — restate facts
    naturally, even when using the same key terms.
- Keep answers helpful, concise, and easy to understand.
- If useful, organize information using bullet points.
- For questions about applying to more than one program, distinguish
    between multiple academic programs and multiple admission/quota categories.
    The prospectus explicitly requires separate applications for each additional
    eligible quota category, but do not claim that applicants can submit one
    application for multiple programs unless the context says so.
- For department-list questions, use department pages and the university home
    page as the authoritative list. Prefer those named department entries over
    historical messages about departments being planned or launched.
- For fee questions, report the application processing and prospectus fees
    exactly as stated in the context: Rs. 2,000 for BS Computer Science
    (non-engineering) programs and Rs. 1,500 for engineering programs. Do not
    present either amount as total tuition or invent other charges. If the user
    asks about fees not stated in the context, direct them to the official fee
    structure page.
- Use all relevant facts in the context, even when the wording differs from the question.
- Do not say that you do not have the information when the context contains related facts
    that can answer the question. State only the supported facts and clearly identify
    any part that is not covered.
- If the context contains no relevant facts at all, say that you do not have that information.
- When information is unavailable, suggest checking the official UET Mardan website.
- Do not make up facts.
- Do not invent admission dates, fees, faculty names, policies, or other university information.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# DOCUMENT RETRIEVAL
# ============================================================

def retrieve_documents(question):

    normalized_question = (
        question.lower()
        .replace("-", " ")
    )
    normalized_question_with_boundaries = f" {normalized_question} "

    if any(term in normalized_question for term in ADMISSION_PROCESS_TERMS):
        return vectordb.similarity_search(
            "undergraduate application procedure online form required documents fee bank account",
            k=4,
            filter={"filename": ADMISSION_PROCESS_FILENAME},
        )

    if any(term in normalized_question for term in ABOUT_UET_TERMS):
        return vectordb.similarity_search(
            "UET Mardan public sector university recognized PEC HEC departments Center Artificial Intelligence",
            k=4,
            filter={"filename": "home.txt"},
        )

    if any(term in normalized_question for term in UNDERGRADUATE_PROGRAM_TERMS):
        targeted_documents = []
        for query, filename in (
            ("bachelor degree programs offered engineering computer science artificial intelligence", "uetm_Admissions_admissionschedule.txt"),
            ("bachelor degrees offered BSc engineering BS programs", CURRENT_PROSPECTUS_FILENAME),
        ):
            targeted_documents.extend(
                vectordb.similarity_search(
                    query,
                    k=4,
                    filter={"filename": filename},
                )
            )
        return targeted_documents[:8]

    if any(term in normalized_question for term in ENTRANCE_TEST_TERMS):
        return vectordb.similarity_search(
            "entrance test engineering programs ETEA Educational Testing Evaluation Agency",
            k=4,
            filter={"filename": CURRENT_PROSPECTUS_FILENAME},
        )

    if any(term in normalized_question for term in MULTI_APPLICATION_TERMS):
        targeted_documents = []
        seen = set()
        for query in (
            "candidates applying for more than one category separate applications",
            "separate application form additional category admission",
            question,
        ):
            for document in vectordb.similarity_search(query, k=4):
                document_key = (
                    document.metadata.get("source_url"),
                    document.page_content,
                )
                if document_key not in seen:
                    seen.add(document_key)
                    targeted_documents.append(document)
        return targeted_documents[:10]

    if any(term in normalized_question for term in DEPARTMENT_LIST_TERMS):
        targeted_documents = []
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
            for document in vectordb.similarity_search(query, **search_kwargs):
                document_key = (
                    document.metadata.get("source_url"),
                    document.page_content,
                )
                if document_key not in seen:
                    seen.add(document_key)
                    targeted_documents.append(document)
        return targeted_documents[:12]

    if any(term in normalized_question for term in FEE_TERMS):
        targeted_documents = []
        seen = set()
        for query, filter_metadata in (
            (
                "undergraduate application processing and prospectus fee engineering computer science",
                {"filename": ADMISSION_PROCESS_FILENAME},
            ),
            ("Fee Structure admission fees", {"filename": "uetm_Admissions_bsfee.txt"}),
            (question, None),
        ):
            search_kwargs = {"k": 3}
            if filter_metadata:
                search_kwargs["filter"] = filter_metadata
            for document in vectordb.similarity_search(query, **search_kwargs):
                document_key = (
                    document.metadata.get("source_url"),
                    document.page_content,
                )
                if document_key not in seen:
                    seen.add(document_key)
                    targeted_documents.append(document)
        return targeted_documents[:8]

    if "pro vice chancellor" in normalized_question:

        exact_url = PRO_VC_URL
        exact_query = "Pro-Vice Chancellor"

    elif (
        "vice chancellor" in normalized_question
        or " vc " in normalized_question_with_boundaries
    ):

        exact_url = VC_URL
        exact_query = "Vice Chancellor"

    elif any(
        term in normalized_question
        for term in (
            "transport",
            "bus service",
            "transportation",
        )
    ):

        exact_url = TRANSPORT_URL
        exact_query = "transport facility for students"

    else:

        exact_url = None

    if exact_url:

        exact_documents = vectordb.similarity_search(
            exact_query,
            k=3,
            filter={
                "source_url": exact_url
            },
        )

        general_documents = vectordb.similarity_search(
            question,
            k=6,
        )

        return (
            exact_documents
            + [
                document
                for document in general_documents
                if document.metadata.get("source_url")
                != exact_url
            ]
        )

    return vectordb.similarity_search(
        question,
        k=6,
    )


# ============================================================
# GEMINI RESPONSE
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def generate_response(prompt, retry=True):

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

    if retry and finish_reason and finish_reason not in ("STOP", "1", "FinishReason.STOP"):
        nudge = (
            prompt
            + "\n\nIMPORTANT: Paraphrase everything in your own words. "
              "Do not reuse phrasing from the context."
        )
        return generate_response(nudge, retry=False)

    return result


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-logo">🎓</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-title">UET Mardan AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        "Your university information assistant"
        "</div>",
        unsafe_allow_html=True,
    )

    # Clear conversation
    if st.button(
        "🗑️  Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    st.markdown(
        '<div class="sidebar-section">🤖 Assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">MODEL</div>
            <div class="info-value">{GEMINI_MODEL}</div>
        </div>

        <div class="info-card">
            <div class="info-label">KNOWLEDGE BASE</div>
            <div class="info-value">UET Mardan Documents</div>
        </div>

        <div class="info-card">
            <div class="info-label">STATUS</div>
            <div class="info-value">🟢 Online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">💡 You can ask about</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            🎓 Admissions & eligibility
        </div>

        <div class="info-card">
            📚 Programs & courses
        </div>

        <div class="info-card">
            💰 Fees & scholarships
        </div>

        <div class="info-card">
            🚌 Transport facilities
        </div>

        <div class="info-card">
            🏛️ University administration
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">🔗 Official Website</div>',
        unsafe_allow_html=True,
    )

    st.link_button(
        "🌐 Visit UET Mardan",
        "https://uetmardan.edu.pk/",
        use_container_width=True,
    )

    st.caption(
        "Answers are generated from the chatbot's indexed UET Mardan knowledge base."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">

        <div class="header-logo">
            🎓
        </div>

        <div>
            <div class="header-title">
                UET Mardan AI Assistant
            </div>

            <div class="header-subtitle">
                Your intelligent guide to UET Mardan
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-card">

            <div class="welcome-title">
                👋 Welcome to UET Mardan AI Assistant
            </div>

            <div class="welcome-text">
                Ask me anything about UET Mardan.
                I can help you find information about
                admissions, programs, fees, university
                facilities, administration, transport,
                and more.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="suggestion-title">✨ Try asking</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="suggestion-card">
                <div class="suggestion-icon">🎓</div>
                <div class="suggestion-text">
                    What programs does UET Mardan offer?
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="suggestion-card">
                <div class="suggestion-icon">💰</div>
                <div class="suggestion-text">
                    Tell me about admission fees.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="suggestion-card">
                <div class="suggestion-icon">🚌</div>
                <div class="suggestion-text">
                    Does UET Mardan provide transport?
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    if role == "user":
        avatar = "🧑‍🎓"
    else:
        avatar = "🎓"

    with st.chat_message(
        role,
        avatar=avatar,
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask about admissions, programs, fees, transport..."
)


if user_input:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message(
        "user",
        avatar="🧑‍🎓",
    ):

        st.markdown(user_input)


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🎓",
    ):

        # Animated typing indicator
        typing_placeholder = st.empty()

        typing_placeholder.markdown(
            """
            <div class="typing-indicator">
                <span>Thinking</span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:

            # ------------------------------------------------
            # RETRIEVE
            # ------------------------------------------------

            documents = retrieve_documents(
                user_input
            )

            context = "\n\n".join(
                document.page_content
                for document in documents
            )[:MAX_CONTEXT_CHARS]

            prompt = PROMPT_TEMPLATE.format(
                context=context,
                question=user_input,
            )

            # ------------------------------------------------
            # GENERATE
            # ------------------------------------------------

            normalized_user_input = user_input.lower()

            if any(term in normalized_user_input for term in UNDERGRADUATE_PROGRAM_TERMS):
                answer = UNDERGRADUATE_PROGRAM_RESPONSE.strip()
            elif any(term in normalized_user_input for term in ENTRANCE_TEST_TERMS):
                answer = ENTRANCE_TEST_RESPONSE
            elif any(term in normalized_user_input for term in ABOUT_UET_TERMS):
                answer = ABOUT_UET_RESPONSE.strip()
            elif any(term in normalized_user_input for term in ADMISSION_PROCESS_TERMS):
                answer = ADMISSION_PROCESS_RESPONSE.strip()
            elif any(term in normalized_user_input for term in FEE_TERMS):
                answer = FEE_RESPONSE.strip()
            else:
                answer = generate_response(
                    prompt
                )

            # Remove typing indicator
            typing_placeholder.empty()

            if not answer:
                answer = (
                    "I couldn't generate an answer right now. "
                    "Please try asking your question again."
                )

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            sources = list(
                {
                    document.metadata.get(
                        "source_url"
                    )
                    for document in documents
                    if document.metadata.get(
                        "source_url"
                    )
                }
            )

            answer_text = answer

            if sources:

                answer_text += (
                    "\n\n---\n\n"
                    "### 📚 Sources\n"
                )

                for source in sources:

                    answer_text += (
                        f"- [{source}]({source})\n"
                    )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            st.markdown(
                answer_text
            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                }
            )


        # ====================================================
        # ERROR HANDLING
        # ====================================================

        except errors.ClientError as error:

            typing_placeholder.empty()

            normalized_user_input = user_input.lower()

            if any(term in normalized_user_input for term in UNDERGRADUATE_PROGRAM_TERMS):
                answer = UNDERGRADUATE_PROGRAM_RESPONSE.strip()
            elif any(term in normalized_user_input for term in ABOUT_UET_TERMS):
                answer = ABOUT_UET_RESPONSE.strip()
            elif any(term in normalized_user_input for term in ADMISSION_PROCESS_TERMS):
                answer = ADMISSION_PROCESS_RESPONSE.strip()
            elif any(term in normalized_user_input for term in FEE_TERMS):

                answer = FEE_RESPONSE.strip()

            elif (
                error.code == 429
                or "quota" in str(error).lower()
                or "resource_exhausted" in str(error).lower()
            ):

                answer = local_quota_fallback(
                    user_input,
                    context,
                )

            else:

                answer = (
                    "⚠️ **Gemini API is temporarily unavailable.**\n\n"
                    "Please try again later."
                )

            st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )


        except Exception as error:

            typing_placeholder.empty()

            answer = (
                "⚠️ **Something went wrong.**\n\n"
                "Please try again in a moment."
            )

            st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        opacity:0.45;
        font-size:11px;
        margin-top:30px;
        padding-bottom:20px;
    ">
        UET Mardan AI Assistant • Powered by RAG + Gemini
    </div>
    """,
    unsafe_allow_html=True,
)