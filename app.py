import os
import json
from dotenv import load_dotenv
import base64
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.storage import RedisStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_elasticsearch import ElasticsearchStore
import streamlit as st
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from base64 import b64decode

load_dotenv()


def display_base64_image(base64_code):
    # Decode the base64 string to binary
    image_data = base64.b64decode(base64_code)
    filename = 'image.jpg'  # You can choose a different filename or make it dynamic
    with open(filename, 'wb') as f:
        f.write(image_data)


def parse_docs(docs):
    """Split base64-encoded images and texts"""
    # print("parse_docs =====> ", docs)
    b64 = []
    text = []
    for doc in docs:
        doc = json.loads(doc)
        print("doc", doc["metadata"].keys())
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    # print("kwargs", kwargs)

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    # print("docs_by_type", docs_by_type)

    context_text = ""

    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            # print("text_element", json.loads(text_element))
            context_text += text_element["text"]

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        # print("docs_by_type", docs_by_type)
        for image in docs_by_type["images"]:
            # print("image", image)
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )


# chain = (
#     {
#         "context": retriever | RunnableLambda(parse_docs),
#         "question": RunnablePassthrough(),
#     }
#     | RunnableLambda(build_prompt)
#     | ChatOpenAI(model="gpt-4o-mini")
#     | StrOutputParser()
# )


def main():

    openai_api_key = st.secrets["OPENAI_API_KEY"]
    es_cloud_id = st.secrets["ES_CLOUD_ID"]
    es_username = st.secrets["ES_USERNAME"]
    es_password = st.secrets["ES_PASSWORD"]
    redis_url = st.secrets["REDIS_URL"]

    elastic_vector_search = ElasticsearchStore(
        es_cloud_id=es_cloud_id,
        index_name="aehtextbook_rag",
        embedding=OpenAIEmbeddings(api_key=openai_api_key),
        es_user=es_username,
        es_password=es_password,
    )

    store = RedisStore(redis_url=redis_url)

    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=elastic_vector_search,
        docstore=store,
        id_key=id_key
    )

    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | ChatOpenAI(model="gpt-4o-mini")
            | StrOutputParser()
        )
    )

    st.set_page_config(page_title="ASK African Economic History")

    st.header("Ask anything about African Economic History 💬")
    # show user input
    user_question = st.text_input(
        "Ask a question about African Ecoonomic History:")
    if user_question:
        response = chain_with_sources.invoke(user_question)

        st.header("Answer", divider=True)
        st.write(response['response'])
        print("\n\nMy reponse comes from this context:")
        contexts = []
        for text in response['context']['texts']:
            context = {}
            context['text'] = text['text']
            context['page_number'] = text['metadata']['page_number']
            context['filename'] = text['metadata']['filename']
            contexts.append(context)
            # print(text['text'][:100])
            # print("Metadata: ", text['metadata'] )
            # print("Page number: ", text['metadata']['page_number'])
            # print("\n" + "-"*50 + "\n")
        # for image in response['context']['images']:
        #     display_base64_image(image)

 

        if contexts:
            st.header("Sources", divider=True)
            for index, context in enumerate(contexts): 
                st.subheader(f"Source #{index + 1} from page #{context['page_number']}", divider=True)
                st.write(f"Document: African Economic History Textbook") 
                st.write(context['text']) 




if __name__ == "__main__":
    main()
