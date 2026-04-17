from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv('.env')

endpoint = "https://google-solutions-2026-resource.openai.azure.com/openai/v1"
deployment_name = "gpt-5.4-mini"
api_key = os.getenv("AZURE_OPENAI_API_KEY")

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)


try:
    completion = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print("SUCCESS OpenAI Client")
    print(completion.choices[0].message)
except Exception as e:
    print("FAIL OpenAI Client:", str(e))

from openai import AzureOpenAI
client2 = AzureOpenAI(
    azure_endpoint="https://google-solutions-2026-resource.openai.azure.com/",
    api_key=api_key,
    api_version="2024-02-01"
)

try:
    completion2 = client2.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print("SUCCESS AzureOpenAI Client")
    print(completion2.choices[0].message)
except Exception as e:
    print("FAIL AzureOpenAI Client:", str(e))
