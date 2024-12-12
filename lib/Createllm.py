from llama_index.llms.ollama import Ollama

def create_llm(model):
    llm = Ollama(model=model,request_timeout=360.0)
    return llm
