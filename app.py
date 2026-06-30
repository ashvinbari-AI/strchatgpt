import streamlit as st
import json
import os
from dotenv import load_dotenv
from google import genai 
from langsmith import wrappers 
from langsmith import traceable
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

JSON_PATH = r"A:\paper_agent\output\loksatta_complete.json"

COLLECTION_NAME = "loksatta"

QDRANT_PATH = "./qdrant_db"

# ----------------------------
# Streamlit Config
# ----------------------------

st.set_page_config(
    page_title="लोकसत्ता सहाय्यक",
    page_icon="📰",
    layout="wide"
)

# ----------------------------
# Load JSON Metadata
# ----------------------------

@st.cache_resource
def load_metadata():

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["extraction_date"], len(data["pages"])


# ----------------------------
# Load Embedding Model
# ----------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "BAAI/bge-m3",
        trust_remote_code=True
    )


# ----------------------------
# Load Qdrant
# ----------------------------

@st.cache_resource
def load_qdrant():

    return QdrantClient(path=QDRANT_PATH)


# ----------------------------
# Gemini Client
# ----------------------------
@st.cache_resource
def load_gemini():

    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    traced_client = wrappers.wrap_gemini(
        gemini_client,
        tracing_extra={
            "tags": [
                "streamlit",
                "marathi-news",
                "rag",
                "gemini"
            ],
            "metadata": {
                "llm": "gemini-2.5-flash",
                "embedding": "BAAI/bge-m3",
                "vectordb": "Qdrant"
            }
        }
    )

    return traced_client
# @st.cache_resource
# def load_gemini():

#     return genai.Client(api_key=GEMINI_API_KEY)


# ----------------------------
# Retrieve Documents
# ----------------------------



@traceable(name="Retrieve Documents")
def retrieve(question, top_k=5):  # Reduced from 10 to 5

    model = load_embedding_model()

    qdrant = load_qdrant()

    query_vector = model.encode(
        question,
        normalize_embeddings=True
    )

    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=top_k
    )

    docs = []

    SIMILARITY_THRESHOLD = 0.60

    for hit in search_result:

        if hit.score < SIMILARITY_THRESHOLD:
            continue

        docs.append({
            "page": hit.payload["page"],
            "text": hit.payload["text"][:800],   # Reduce context
            "score": hit.score
        })

    return docs

# ----------------------------
# Ask Gemini
# ----------------------------

@traceable(name="Marathi Newspaper RAG")
def rag_pipeline(question, date):

    retrieved = retrieve(question)

    answer = ask_gemini(
        question,
        retrieved,
        date
    )

    return answer, retrieved
@traceable(name="Ask Gemini")
def ask_gemini(question, retrieved_docs, date):

    if not retrieved_docs:
        return "माफ करा, या विषयाची माहिती उपलब्ध नाही."

    context = ""

    pages = sorted(set(item["page"] for item in retrieved_docs))

    for item in retrieved_docs:

        context += f"""

पान {item['page']}:

{item['text']}

"""

    prompt = f"""
तुम्ही लोकसत्ता वृत्तपत्र सहाय्यक आहात.

फक्त दिलेल्या संदर्भाचा वापर करा.

जर माहिती उपलब्ध नसेल तर उत्तर द्या:

"माफ करा, ही माहिती लोकसत्ता वर्तमानपत्रात उपलब्ध नाही."

उत्तर मराठीत द्या.

उत्तर 100 शब्दांमध्ये द्या.

======================

दिनांक

{date}

======================

संदर्भ

{context}

======================

प्रश्न

{question}

======================

उत्तर
"""

    client = load_gemini()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0.2,
            "top_p": 0.8,
            "max_output_tokens": 300
        }
    )

    answer = response.text

    answer += "\n\n📄 स्रोत पृष्ठे : "

    answer += ", ".join(map(str, pages))

    return answer
# ----------------------------
# Main App
# ----------------------------

def main():

    st.title("📰 लोकसत्ता बातमी सहाय्यक")

    date, total_pages = load_metadata()

    st.info(
        f"📅 दिनांक : {date} | 📄 पाने : {total_pages}"
    )

    if "messages" not in st.session_state:

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"🙏 नमस्कार! मी {date} च्या लोकसत्ता बातम्यांचा AI सहाय्यक आहे."
            }
        ]

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

    question = st.chat_input("प्रश्न विचारा...")

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("🔍 बातम्यांमध्ये शोध सुरू आहे..."):

                retrieved = retrieve(question)

                answer = ask_gemini(
                    question,
                    retrieved,
                    date
                )

                st.markdown(answer)

                with st.expander("📄 Retrieved Pages"):

                    for item in retrieved:

                        st.write(
                            f"Page {item['page']} | Similarity : {item['score']:.4f}"
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


if __name__ == "__main__":

    main()