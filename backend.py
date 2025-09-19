import torch
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
from transformers import BitsAndBytesConfig

def hf_login(token: str):
    login(token=token)

def get_embed_model():
    return LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

def get_llm(system_prompt: str):
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    return HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        system_prompt=system_prompt,
        tokenizer_name="meta-llama/Llama-3.2-1B-Instruct",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        device_map="auto",
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": quant_config,
            "pad_token_id": 128001
        }
    )

def load_documents(directory: str):
    return SimpleDirectoryReader(directory).load_data()

def build_index_and_engine(documents):
    index = VectorStoreIndex.from_documents(documents)
    return index, index.as_query_engine()

def main():
    hf_login(token="") # "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"
    system_prompt = (
        "You are a Q&A assistant. Your goal is to answer questions based on the given documents."
    )
    embed_model = get_embed_model()
    llm = get_llm(system_prompt)

    # Apply global Settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    Settings.context_window = 4096

    print("Q&A Assistant is ready! Type your question (or 'exit' to quit):")

    query_engine = None
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        else:
            documents = load_documents("files")
            _, query_engine = build_index_and_engine(documents)
        response = query_engine.query(user_query)
        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    main()
