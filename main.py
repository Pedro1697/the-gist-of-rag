import os 

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Inicializing components...")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(model="gemini-3-flash",api_key=GOOGLE_API_KEY)

vectorestore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], 
    embedding=embeddings
)

retriever = vectorestore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}

    Question: {question}

    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retireved docs into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """
    simple retireval chain without LCEL.
    Manually retireves documents, formats them, and generates a response.

    Limitations:
    - Manually step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve revelant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLm with context and question
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content

if __name__ == "__main__":
    print("Retrieving...")

    #Query
    query = "What is Pinecone in machine learning?"

     # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")

