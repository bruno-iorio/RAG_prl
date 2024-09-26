from llama_index.llms.ollama import Ollama

def create_llm():
    llm = Ollama(model='llama3.1',request_timeout=360.0)
    return llm
