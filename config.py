import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Centralized configuration for InsightForge"""

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    app_title = "InsightForge - The Business Intelligence Assistant"
    layout = "wide"

    default_csv_path = "sales_data.csv"
    sample_data_size = 2500

    model_name = "gpt-4o-mini"
    llm_temperature = 0.1
    max_tokens = 2000
    chunk_size = 1000
    chunk_overlap = 200

    chroma_path = "./chroma_db"
    EMBED_MODEL = "text-embedding-3-small"

    chart_height = 400
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    sample_products = ['Widget A', 'Widget B', 'Widget C', 'Widget D']
    sample_regions = ['North', 'South', 'East', 'West']
    sample_genders = ['Male', 'Female']

    @classmethod
    def get_chart_config(cls):
        return {
            'height': cls.chart_height,
            'color_palette': cls.color_palette
        }