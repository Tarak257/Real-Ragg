import time
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

REWRITER_MODES = [
    "Direct (No Rewriting)",
    "Multi-Query Expansion",
    "HyDE (Hypothetical Document Embedding)",
    "Step-Back Prompting"
]

def _invoke_with_retry(llm: ChatGoogleGenerativeAI, prompt_str: str, retries: int = 1):
    fallback_models = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    models_to_try = [getattr(llm, "model", "gemini-3.6-flash")] + [
        m for m in fallback_models if m != getattr(llm, "model", "")
    ]
    
    last_err = None
    for model_name in models_to_try:
        try:
            active_llm = ChatGoogleGenerativeAI(model=model_name)
            return active_llm.invoke(prompt_str)
        except Exception as e:
            last_err = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "404" in str(e):
                time.sleep(1)
                continue
            else:
                raise e
    raise last_err


def _extract_text_from_resp(resp) -> str:
    if hasattr(resp, "content"):
        content = resp.content
    else:
        content = resp

    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(getattr(item, "text", item))
            for item in content
        )
    return str(content)


def rewrite_query(
    query: str,
    mode: str,
    llm: ChatGoogleGenerativeAI
) -> Dict[str, Any]:
    """
    Transforms the user input query based on the selected query rewriting strategy.
    
    Returns a dictionary containing:
      - mode: Strategy used
      - original_query: Input query
      - search_queries: List of queries to execute against retrieval engines
      - hyde_document: Optional hypothetical passage generated for HyDE
      - step_back_query: Optional step-back query generated
    """
    clean_query = query.strip()
    result = {
        "mode": mode,
        "original_query": clean_query,
        "search_queries": [clean_query],
        "hyde_document": None,
        "step_back_query": None
    }
    
    if mode == "Direct (No Rewriting)" or not clean_query:
        return result
        
    try:
        if mode == "Multi-Query Expansion":
            prompt = ChatPromptTemplate.from_template("""
You are an AI language model assistant for RAG systems.
Your task is to generate 3 different versions/perspectives of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user overcome distance limitations of distance-based similarity search.
Provide these alternative questions separated by newlines. Do NOT add numbering or bullet points.

Original Question: {question}
""")
            resp = _invoke_with_retry(llm, prompt.format(question=clean_query))
            raw_text = _extract_text_from_resp(resp)

            lines = [line.strip("- *1234567890. ") for line in raw_text.split("\n") if line.strip()]
            search_queries = [clean_query] + [l for l in lines if l and l.lower() != clean_query.lower()]
            result["search_queries"] = search_queries[:4]

        elif mode == "HyDE (Hypothetical Document Embedding)":
            prompt = ChatPromptTemplate.from_template("""
Please write a concise hypothetical passage or excerpt from an authoritative document that directly answers the question below.
Do NOT include preamble, introductory remarks, or explanations. Write only the passage text itself.

Question: {question}
""")
            resp = _invoke_with_retry(llm, prompt.format(question=clean_query))
            hyde_doc = _extract_text_from_resp(resp)
            
            result["hyde_document"] = hyde_doc.strip()
            result["search_queries"] = [clean_query, hyde_doc.strip()]

        elif mode == "Step-Back Prompting":
            prompt = ChatPromptTemplate.from_template("""
You are an expert at step-back prompting.
Given the user's specific query, generate a single higher-level, broader concept question that provides general background context.
Output ONLY the step-back question without preamble.

Specific Query: {question}
""")
            resp = _invoke_with_retry(llm, prompt.format(question=clean_query))
            step_back = _extract_text_from_resp(resp)
            step_back_clean = step_back.strip()
            
            result["step_back_query"] = step_back_clean
            result["search_queries"] = [clean_query, step_back_clean]

    except Exception as e:
        print(f"Warning: Query rewriter failed with error: {e}. Falling back to original query.")
        result["search_queries"] = [clean_query]
        
    return result

