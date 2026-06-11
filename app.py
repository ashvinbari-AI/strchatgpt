import streamlit as st
import json
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

JSON_PATH = r"A:\paper_agent\output\loksatta_complete.json"


st.set_page_config(
    page_title="लोकसत्ता सहाय्यक",
    page_icon="📰",
    layout="wide"
)


@st.cache_resource
def load_data():

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data["pages"]

    texts = []

    for page in pages:

        texts.append(
            {
                "page": page["page_number"],
                "text": page["text"]
            }
        )

    return texts, data["extraction_date"]


@st.cache_resource
def build_index(texts):

    corpus = [x["text"] for x in texts]

    vectorizer = TfidfVectorizer(
        stop_words=None,
        max_features=10000
    )

    vectors = vectorizer.fit_transform(corpus)

    return vectorizer, vectors


def retrieve(question,
             texts,
             vectorizer,
             vectors,
             top_k=3):

    q_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        q_vector,
        vectors
    )[0]

    indices = scores.argsort()[-top_k:][::-1]

    results = []

    for idx in indices:

        results.append(
            {
                "page": texts[idx]["page"],
                "text": texts[idx]["text"],
                "score": scores[idx]
            }
        )

    return results


@st.cache_resource
def load_gemini():

    return genai.Client(
        api_key=GEMINI_API_KEY
    )


def ask_gemini(question,
               retrieved_pages,
               date):

    context = ""

    pages_used = []

    for item in retrieved_pages:

        pages_used.append(str(item["page"]))

        context += f"""

पान {item['page']}:

{item['text']}

"""

    prompt = f"""
तुम्ही लोकसत्ता वृत्तपत्र सहाय्यक आहात.

दिनांक:
{date}

संदर्भ:

{context}

प्रश्न:
{question}

नियम:

1. फक्त दिलेल्या संदर्भातील माहिती वापरा.

2. माहिती नसेल तर म्हणा:
"माफ करा, ही माहिती आजच्या बातमीत उपलब्ध नाही."

3. अंदाज लावू नका.

4. उत्तर मराठीत द्या.

5. शेवटी वापरलेली पान क्रमांक सांगा.

उत्तर:
"""

    client = load_gemini()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


def main():

    st.title("📰 लोकसत्ता बातमी सहाय्यक")

    texts, date = load_data()

    st.info(
        f"📅 दिनांक: {date} | "
        f"📄 पाने: {len(texts)}"
    )

    vectorizer, vectors = build_index(texts)

    if "messages" not in st.session_state:

        st.session_state.messages = []

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content":
                f"🙏 नमस्कार! मी {date} च्या लोकसत्ता बातम्यांचा सहाय्यक आहे."
            }
        )

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

    question = st.chat_input(
        "प्रश्न विचारा..."
    )

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

            with st.spinner("🔍 बातमीत शोधत आहे..."):

                retrieved = retrieve(
                    question,
                    texts,
                    vectorizer,
                    vectors
                )

                answer = ask_gemini(
                    question,
                    retrieved,
                    date
                )

                st.markdown(answer)

                with st.expander("📄 वापरलेली पाने"):

                    for item in retrieved:

                        st.write(
                            f"पान {item['page']} "
                            f"(score={item['score']:.2f})"
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


if __name__ == "__main__":

    main()