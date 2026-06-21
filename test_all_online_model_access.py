import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

MODEL_TESTS = [
    {
        "name": "OpenAI GPT-OSS 20B",
        "model": "openai/gpt-oss-20b",
        "api_key": os.getenv("NVIDIA_API_KEY_OPENAI_GPT_OSS_20B"),
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
    },
    {
        "name": "Google Gemma 4 31B IT",
        "model": "google/gemma-4-31b-it",
        "api_key": os.getenv("NVIDIA_API_KEY_GOOGLE_GEMMA_4_31B_IT"),
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 512,
    },
    {
        "name": "NVIDIA Nemotron 3 Super 120B",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "api_key": os.getenv("NVIDIA_API_KEY_NEMOTRON_3_SUPER"),
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 512,
    },
    {
        "name": "Mistral Nemotron",
        "model": "mistralai/mistral-nemotron",
        "api_key": os.getenv("NVIDIA_API_KEY_MISTRAL_NEMOTRON"),
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 512,
    },
    {
        "name": "Mixtral 8x7b Instruct",
        "model": "mistralai/mixtral-8x7b-instruct-v0.1",
        "api_key": os.getenv("NVIDIA_API_KEY_MIXTRAL_8X7B"),
        "temperature": 0.5,
        "top_p": 1.0,
        "max_tokens": 512,
    },
]

PROMPT = "Write a short 2-sentence introduction about LangChain."

for test in MODEL_TESTS:
    print("\n" + "=" * 80)
    print(f"Testing: {test['name']} ({test['model']})")

    if not test["api_key"]:
        print("  ERROR: missing API key for this model.")
        continue

    client = ChatNVIDIA(
        model=test["model"],
        api_key=test["api_key"],
        temperature=test["temperature"],
        top_p=test["top_p"],
        max_completion_tokens=test["max_tokens"],
        model_kwargs={"enable_thinking": True},
    )

    response = client.invoke([{"role": "user", "content": PROMPT}])
    print("Result:")
    print(response.content)