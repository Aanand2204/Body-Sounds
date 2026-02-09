import os
import json
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Note: Heavy LangChain/LangGraph imports moved inside build_heartbeat_agent
# to ensure zero-cost package import.

# Global cache for embeddings to avoid reloading in every agent instance
_embeddings_cache = None

def get_embeddings():
    """Load and return HuggingFace embeddings model (manual caching)."""
    global _embeddings_cache
    if _embeddings_cache is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_cache = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_cache

def build_retriever(json_path: str, sound_type: str = "heart"):
    """Load JSON, split into chunks, and return a retriever."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    
    with open(json_path, "r") as f:
        report_data = json.load(f)

    # Convert JSON to pretty string
    report_text = json.dumps(report_data, indent=2)

    # Wrap into LangChain Document
    docs = [Document(page_content=report_text, metadata={"source": f"{sound_type}_report"})]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    doc_chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = get_embeddings()

    # Vectorstore + retriever
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore.as_retriever()


# --- Tool wrapper ---
def make_retriever_tool(retriever, sound_type: str):
    from langchain_core.tools import tool
    
    @tool
    def sound_retriever_tool(query: str) -> str:
        """
        Retrieve information from the sound analysis report.
        Call this tool whenever a user asks about specific analysis results.
        Returns the content of relevant documents.
        """
        docs = retriever.invoke(query)
        # Combine docs into a single string for valid tool output
        return "\n\n".join([d.page_content for d in docs])

    return sound_retriever_tool


# --- Specialized Prompts ---

PROMPTS = {
    "heart": """
You are HeartbeatAnalysisAgent, an expert assistant for analyzing heartbeat recordings.
The following is the analysis result for the current recording:

--- PATIENT INFO ---
{patient_info}

--- ANALYSIS REPORT ---
{report_summary}
--- END OF DATA ---

Your role:
- Answer user questions using the data provided above.
- If (and ONLY if) the data above is missing details, you can use 'sound_retriever_tool'.
- Explain heart rate metrics: BPM, heart rate variability (HRV), inter-beat intervals (IBI).
- Interpret valve sounds (S1/S2), murmurs, and signal quality (SNR).

**Expert Knowledge (How we detect heartbeats):**
- We use a 20-500Hz bandpass filter to isolate heart sounds.
- We extract the energy envelope using a 40ms sliding window.
- Peaks (S1/S2 candidates) are detected on the envelope using adaptive thresholds (percentile-based height and prominence).
- S1 and S2 are identified based on their sequence; the S1/S2 ratio compares their relative amplitudes.

- If asked about medical interpretation, provide general information only, not medical advice.
- Keep explanations clear and concise.
""",
    "bowel": """
You are BowelSoundAnalysisAgent, an expert assistant for analyzing bowel sound recordings.
The following is the analysis result for the current recording:

--- PATIENT INFO ---
{patient_info}

--- ANALYSIS REPORT ---
{report_summary}
--- END OF DATA ---

Your role:
- Answer user questions using the data provided above.
- If (and ONLY if) the data above is missing details, you can use 'sound_retriever_tool'.
- Explain event detection metrics: events detected, event rate (per minute), event intervals.
- Interpret bowel activity patterns and spectral features (centroid, band energy).

**Expert Knowledge (How we detect bowel sounds):**
- We use a 100-1000Hz bandpass filter.
- Events are detected by identifying peaks in the energy envelope that exceed the 60th percentile height and a specific prominence threshold.

- If asked about medical interpretation, provide general information only, not medical advice.
- Keep explanations clear and concise.
""",
    "lung": """
You are LungSoundAnalysisAgent, an expert assistant for analyzing respiratory sound recordings.
The following is the analysis result for the current recording:

--- PATIENT INFO ---
{patient_info}

--- ANALYSIS REPORT ---
{report_summary}
--- END OF DATA ---

Your role:
- Answer user questions using the data provided above.
- If (and ONLY if) the data above is missing details, you can use 'sound_retriever_tool'.
- Explain breathing rate, respiratory conditions (COPD, pneumonia, bronchitis, asthma, etc.).
- Interpret adventitious sounds: wheeze index, crackle index, spectral features.

**Expert Knowledge (How we detect breaths):**
- We use a bandpass filter and envelope extraction.
- Breath cycles are detected by finding peaks in the envelope with a minimum distance of 0.8 seconds and prominence above the standard deviation.

- If asked about medical interpretation, provide general information only, not medical advice.
- Keep explanations clear and concise.
"""
}

# --- Generic Builder ---

def build_generic_agent(sound_type: str, json_path: str, patient_info: Optional[dict] = None, groq_api_key: Optional[str] = None, hf_token: Optional[str] = None):
    from langchain_groq import ChatGroq
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool
    
    # Format patient info
    if patient_info:
        p_str = json.dumps(patient_info, indent=2)
    else:
        p_str = "No patient information provided."

    # Load report data directly for faster context injection
    try:
        with open(json_path, "r") as f:
            report_data = json.load(f)
        report_summary = json.dumps(report_data, indent=2)
    except Exception as e:
        report_summary = f"Error loading report: {str(e)}"
    
    # Set HF token if provided
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    # Initialize Groq Chat Model
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY")
    )

    # Build retriever here (in the main thread)
    retriever = build_retriever(json_path, sound_type)

    # Define the tool
    @tool
    def sound_retriever_tool(query: str) -> str:
        """
        Retrieve specific information from the report JSON.
        Call this tool IF and ONLY IF the provided report summary or patient info is insufficient.
        """
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])

    tools = [sound_retriever_tool]
    
    # Get the specialized prompt
    system_message = PROMPTS.get(sound_type, PROMPTS["heart"]).format(
        report_summary=report_summary,
        patient_info=p_str
    )

    # Create the agent
    agent = create_react_agent(llm, tools, prompt=system_message)
    return agent


# --- Exported Functions ---

def build_heartbeat_agent(json_path: str, patient_info: Optional[dict] = None, groq_api_key: Optional[str] = None, hf_token: Optional[str] = None):
    return build_generic_agent("heart", json_path, patient_info, groq_api_key, hf_token)

def build_bowel_agent(json_path: str, patient_info: Optional[dict] = None, groq_api_key: Optional[str] = None, hf_token: Optional[str] = None):
    return build_generic_agent("bowel", json_path, patient_info, groq_api_key, hf_token)

def build_lung_agent(json_path: str, patient_info: Optional[dict] = None, groq_api_key: Optional[str] = None, hf_token: Optional[str] = None):
    return build_generic_agent("lung", json_path, patient_info, groq_api_key, hf_token)

