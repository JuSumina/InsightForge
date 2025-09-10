import sys, platform
print("Python:", sys.version)
print("Platform:", platform.platform())
import pydantic, pydantic_core
print("Pydantic:", pydantic.__version__, "Core:", pydantic_core.__version__)
print("Core file:", getattr(pydantic_core, "__file__", "N/A"))
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
print("LangChain-OpenAI imports OK")