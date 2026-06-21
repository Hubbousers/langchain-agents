import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    information = """
Elon Musk is a technology entrepreneur known for Tesla and SpaceX.
He has led major advances in electric vehicles, reusable rockets,
and AI research through companies like Tesla, SpaceX, Neuralink,
and xAI.
"""

    summary_template = """
Given the information {information} about a topic/person, create:
1. A short summary
2. Two interesting facts about them
"""

    summary_prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOllama(
        model="phi3:mini",
        temperature=1.0,
        top_p=0.95,
        max_completion_tokens=16000,
        model_kwargs={"enable_thinking": True},
    )
    # Chain the prompt and the LLM together and invoke it with the input information
    # Note: the key variable inside the input dictionary must match the input variable name in the prompt template
    chain = summary_prompt | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()