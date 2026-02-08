from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def build_rag_chain(llm, retriever):
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the question using chat history if required."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer using the provided context only.\n\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_retriever, qa_chain)
