from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
try:
    from langchain.chains import ConversationalRetrievalChain
except Exception:
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any, List, Optional
import logging
import json
from config import Config

logger = logging.getLogger(__name__)

try:
    from langchain.evaluation.qa import QAEvalChain
    EVAL_AVAILABLE = True
except Exception:
    QAEvalChain = None
    EVAL_AVAILABLE = False

class PromptTemplates:
    """Collection of prompt templates for different AI use cases"""
    
    @staticmethod
    def get_analysis_prompt() -> PromptTemplate:
        """Business analysis prompt template"""
        template = """
        You are InsightForge, an expert Business Intelligence Assistant analyzing sales data.

        You have:
        - Retrieved context from a knowledge base (may be sparse or purely numeric)
        - The running chat history
        - A new user question

        Context: {context}
        Chat history: {chat_history}
        Question: {question}
        
        Please provide:
        1. Key insights from the data
        2. Trends and patterns identified
        3. Actionable business recommendations
        4. Potential risks or opportunities

        Format:
        - Use normal spaces between words (e.g., "1.38 million", not "1.38million").
        - Use Markdown headings/bullets when helpful.
        - Start every subparagraph from a new line (e.g. - Economic downturns 
                                                         - Increased competition, not - Economic downturns - Increased competition).
        
        Make your response comprehensive yet easy to understand for business stakeholders.
        
        Response:
        """
        return PromptTemplate(
            template=template,
            input_variables=["question", "context", "chat_history"]
        )
    
    @staticmethod
    def get_summary_prompt() -> PromptTemplate:
        """Executive summary prompt template"""
        template = """
        As InsightForge, provide a comprehensive executive summary of the business performance data below:
        
        Data Summary: {data_summary}
        
        Please structure your response as follows:
        1. **Executive Summary** - Key highlights in 2-3 sentences
        2. **Performance Metrics** - Most important numbers and trends
        3. **Strategic Insights** - What the data reveals about business performance
        4. **Recommendations** - Top 3 actionable recommendations

        Format:
        - Use normal spaces between words (e.g., "1.38 million", not "1.38million").
        - Use Markdown headings/bullets when helpful.
        - Start every subparagraph from a new line (e.g. - Economic downturns 
                                                         - Increased competition, not - Economic downturns - Increased competition).
        
        Keep the tone professional but accessible for C-level executives.
        
        Summary:
        """
        return PromptTemplate(template=template, input_variables=["data_summary"])

class CustomRetriever:
    """Custom retriever for business data insights"""
    
    def __init__(self, processed_data: Dict[str, Any], api_key: str):
        self.processed_data = processed_data
        self.api_key = api_key
        self.vectorstore = None
        self._create_knowledge_base()
    
    def _create_knowledge_base(self):
        """Create searchable knowledge base from processed data using Chroma."""
        # Build documents from your processed_data dict
        docs = []
        for category, data in self.processed_data.items():
            text = f"Category: {category}\n" + self._dict_to_text(data)
            docs.append(Document(page_content=text, metadata={"category": category}))

        # Embeddings
        embeddings = OpenAIEmbeddings(api_key=self.api_key, model=Config.EMBED_MODEL)

        # Split docs
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.chunk_size, 
            chunk_overlap=Config.chunk_overlap
        )
        split_docs = splitter.split_documents(docs)

        # Create (or reopen) a Chroma collection that persists to disk
        if getattr(Config, "vectorstore_backend", "chroma") == "chroma":
            import os
            os.environ["ANONYMIZED_TELEMETRY"] = "false"
            os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
            import chromadb
            from langchain_community.vectorstores import Chroma

            client = chromadb.PersistentClient(path=Config.chroma_path)
            self.vectorstore = Chroma.from_documents(
                split_docs,
                embeddings,                 # pass embeddings so Chroma doesn't try to use its own
                client=client,
                collection_name="insightforge",
            )
            logger.info("Knowledge base created using Chroma (persistent)")

        # --- optional fallback to SKLearn if you want it ---
        else:
            from langchain_community.vectorstores import SKLearnVectorStore
            self.vectorstore = SKLearnVectorStore.from_documents(split_docs, embeddings)
            logger.info("Knowledge base created using SKLearnVectorStore")
    
    def _dict_to_text(self, data, indent=0):
        """Convert dictionary to readable text"""
        text = []
        for key, value in data.items():
            if isinstance(value, dict):
                text.append("  " * indent + f"{key}:")
                text.append(self._dict_to_text(value, indent + 1))
            else:
                text.append("  " * indent + f"{key}: {value}")
        return "\n".join(text)
    
    def retrieve_relevant_data(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant data based on query"""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)

class RAGSystem:
    """Complete RAG system for business insights"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = ChatOpenAI(
            model=Config.model_name,
            temperature=Config.llm_temperature,
            api_key=api_key,
            max_tokens=Config.max_tokens
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        self.retriever = None
        self.conversational_chain = None
        
    def setup_rag_system(self, processed_data: Dict[str, Any]):
        """Setup the RAG system with business data"""
        try:
            self.retriever = CustomRetriever(processed_data, self.api_key)
            self.conversational_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever.vectorstore.as_retriever(
                    search_kwargs={"k": 8}
                ),
                memory=self.memory,
                return_source_documents=False,
                combine_docs_chain_kwargs={"prompt": PromptTemplates.get_analysis_prompt()},
            )
            logger.info("RAG system setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up RAG system: {e}")
            raise

    def generate_insights(self, question: str) -> str:
        """Generate insights based on question - main method for all interactions"""
        if not self.conversational_chain:
            raise ValueError("RAG system not setup. Call setup_rag_system() first.")
        
        try:
            result = self.conversational_chain.invoke({"question": question})
            return result.get("answer", "")
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def generate_summary(self, data_summary: str) -> str:
        """Generate executive summary"""
        try:
            prompt = PromptTemplates.get_summary_prompt()
            chain = LLMChain(llm=self.llm, prompt=prompt)
            output = chain.invoke({"data_summary": data_summary})
            
            if isinstance(output, dict):
                text = output.get("text") or output.get("result") or output.get("output") or ""
            else:
                text = str(output)
            return text
        
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"

class ModelEvaluator:
    """Evaluate the RAG model performance"""
    
    def __init__(self, rag_system: 'RAGSystem'):
        if not EVAL_AVAILABLE:
            raise RuntimeError("Model evaluation is unavailable with this LangChain build.")
        self.rag_system = rag_system
        self.eval_chain = QAEvalChain.from_llm(rag_system.llm)
    
    def evaluate_model(self, test_questions: List[str]) -> Dict[str, Any]:
        """Evaluate model with test questions"""
        results = []
        
        for question in test_questions:
            try:
                prediction = self.rag_system.generate_insights(question)
                results.append({
                    'question': question,
                    'prediction': prediction,
                    'evaluation': 'Generated successfully'
                })
            except Exception as e:
                results.append({
                    'question': question,
                    'prediction': '',
                    'evaluation': f'Error: {str(e)}'
                })
        
        return {
            'total_questions': len(test_questions),
            'successful_responses': len([r for r in results if 'Error' not in r['evaluation']]),
            'results': results
        }