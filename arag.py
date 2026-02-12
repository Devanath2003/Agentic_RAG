import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool  # ✅ Removed ScrapeWebsiteTool
from crewai import Agent, Task, Crew, LLM

# 1. Load Environment Variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY2 = os.getenv("GROQ_API_KEY2")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not GROQ_API_KEY or not SERPER_API_KEY:
    raise ValueError("Please set GROQ_API_KEY and SERPER_API_KEY in your .env file.")

# 2. Initialize LLMs

# Main LLM for Routing and Final Answer (Deterministic)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=500,
    groq_api_key=GROQ_API_KEY,
    max_retries=2,
)

# Crew LLM — tool-use fine-tuned model, low max_tokens to stay within TPM
crew_llm_model = LLM(
    model="groq/llama-3.3-70b-versatile",  # ✅ Fine-tuned for tool use
    api_key=GROQ_API_KEY2,
    max_tokens=200,   # ✅ Reduced — agents only need short outputs
    temperature=0.7,
    is_litellm=True
)

# 3. Router Function
def check_local_knowledge(query, context):
    """
    Determines if the local context is sufficient to answer the query.
    Returns True if yes, False if no.
    """
    prompt = '''Role: Question-Answering Assistant

    Task: Determine whether the system can answer the user's question based on the provided text.

    Instructions:
    - Analyze the text and identify if it contains the necessary information.
    - Response must be a single word: "Yes" or "No".

    Examples:
    Input:
        Text: There are 14 districts in Kerala.
        User Question: How many districts are there in Kerala?
    Expected Output:
        Answer: Yes
    
    Input:
        Text: Peacock is the national bird of India.
        User Question: What is the national bird of China?
    Expected Output:
        Answer: No
    
    Now analyze this:
    Input:
        User Question: {query}
        Text: {text}
    Output:'''

    formatted_prompt = prompt.format(text=context, query=query)
    response = llm.invoke(formatted_prompt)
    answer = response.content.strip().lower()
    
    return 'yes' in answer

# 4. Web Search Crew Setup — Search only, no scraping
def setup_web_search_crew():
    """
    Configures and returns a single-agent CrewAI crew for web searching.
    Scraper removed to reduce token usage — Serper snippets are sufficient.
    """
    search_tool = SerperDevTool(n_results=3)  # ✅ Only 3 results instead of 10

    web_search_agent = Agent(
        role="Expert Web Search Agent",
        goal="Search for information about the topic and summarize findings from search results",
        backstory="An expert at finding and summarizing information from web search results.",
        tools=[search_tool],
        verbose=True,
        llm=crew_llm_model
    )

    search_task = Task(
        description=(
            "Search for information about '{topic}'. "
            "Summarize the key facts and concepts from the search results concisely."
        ),
        expected_output=(
            "A concise summary of the most relevant information about '{topic}' "
            "based on the search results."
        ),
        agent=web_search_agent,
    )

    crew = Crew(
        agents=[web_search_agent],
        tasks=[search_task],
        verbose=True,
    )

    return crew

def get_web_content(query):
    print(f"Searching the web for: {query}")
    crew = setup_web_search_crew()
    result = crew.kickoff(inputs={"topic": query})
    return result.raw

# 5. Vector DB Functions
def setup_vector_db(pdf_path):
    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=50, 
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)
    print("Vector database created successfully")
    return vector_db

def get_local_content(vector_db, query):
    docs = vector_db.similarity_search(query, k=5)
    context = " ".join([doc.page_content for doc in docs])
    return context

# 6. Answer Generation
def generate_final_answer(context, query):
    messages = [
        (
            "system",
            "You are a helpful assistant. Use the provided context to answer the user's question accurately. "
            "If the context doesn't contain enough information, say so clearly."
        ),
        ("system", f"Context: {context}"),
        ("human", query),
    ]
    response = llm.invoke(messages)
    return response.content

# 7. Main Pipeline
def process_query(query, vector_db):
    print(f"\nProcessing query: {query}")
    
    # Step 1: Retrieve from local documents
    print("! Checking local documents...")
    local_context = get_local_content(vector_db, query)

    # Step 2: Router Decision
    can_answer_locally = check_local_knowledge(query, local_context)
    print(f"Router Decision: Can answer locally? -> {can_answer_locally}")

    if can_answer_locally:
        print("! Retrieving from local documents..")
        context = local_context
        source = "LOCAL DOCUMENTS"
    else:
        print("! Local context insufficient. Searching the web..")
        context = get_web_content(query)
        source = "WEB SEARCH"

    print(f"\nRetrieved context from {source}.")
    print("-> Generating Final answer...!\n")
    
    answer = generate_final_answer(context, query)
    return answer, source

# 8. Entry Point
def main():
    pdf_path = r"C:\Users\devan\Desktop\Agentic_RAG\Basic-Biology-an-introduction.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    print("Initializing Agentic RAG system..\n")

    print("Step 1: Setting up vector database..")
    vector_db = setup_vector_db(pdf_path)

    query = input("Enter the query: ")
    answer,source = process_query(query,vector_db)

    print(f"{'='*60}")
    print(f"FINAL ANSWER (Source: {source}): ")
    print(f"{'='*60}")
    print(f"\n{answer}\n")
    print(f"{'='*60}\n\n")

    # queries = [
    #     "What is an organism?", 
    #     "Tell me latest Arsenal vs Manchester United score", 
    # ]

    # for query in queries:
    #     answer, source = process_query(query, vector_db)
        
    #     print(f"{'='*60}")
    #     print(f"FINAL ANSWER (Source: {source}): ")
    #     print(f"{'='*60}")
    #     print(f"\n{answer}\n")
    #     print(f"{'='*60}\n\n")

if __name__ == '__main__':
    main()