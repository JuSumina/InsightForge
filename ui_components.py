import streamlit as st
from typing import Optional, Tuple, Any
import tempfile
import os
import json
import logging
from data_handler import DataProcessor
from visualizations import DataVisualizer
from config import Config
import re
from hashlib import md5

logger = logging.getLogger(__name__)

try:
    from ai_system import RAGSystem, ModelEvaluator
    AI_AVAILABLE = True
except Exception as e:
    AI_AVAILABLE = False
    logger.warning(f"AI features disabled: {e}")

def _normalize_llm_text(text: str) -> str:
    import re
    if not isinstance(text, str):
        text = str(text)

    # Strip zero-width and odd separators
    text = re.sub(r'[\u200b-\u200f\u2060\uFEFF\u2028\u2029]', '', text)

    # Collapse excessive newlines: 3+ -> 2 (paragraph), single newlines between words -> space
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Light punctuation/space tidy
    text = re.sub(r',(?!\s)', ', ', text)
    text = re.sub(r'\.(?!\s|[\d])', '. ', text)
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)  # 123ABC -> 123 ABC
    text = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)  # ABC123 -> ABC 123

    # Collapse runs of spaces/tabs
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'[ \t]+\n', '\n', text)

    return text.strip()

def _to_text(maybe) -> str:
    if isinstance(maybe, dict):
        return maybe.get("result") or maybe.get("text") or maybe.get("output") or json.dumps(maybe, ensure_ascii=False)
    return str(maybe)

def _hash_processed(data: dict) -> str:
    # Stable hash of the processed_data dict (handles non-serializable types)
    try:
        payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        payload = json.dumps(data, default=str).encode("utf-8")
    return md5(payload).hexdigest()

class StreamlitApp:
    """Streamlit user interface for InsightForge"""
    
    def __init__(self):
        self._setup_page()
    
    def _setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title=Config.app_title,
            layout=Config.layout,
            initial_sidebar_state="expanded"
        )
    
    def render_header(self):
        """Render application header"""
        st.title("InsightForge")
        st.subheader("AI-Powered Business Intelligence Assistant")
        st.markdown("Transform your sales data into actionable business insights")
        st.markdown("---")
    
    def render_sidebar(self) -> Tuple[str, Optional[Any], bool, bool, bool]:
        """Render sidebar configuration"""
        st.sidebar.title("Configuration")
        
        st.sidebar.subheader("Data Source")
        data_source = st.sidebar.radio(
            "Choose your data source:",
            ["Upload CSV File", "Use Sample Data"],
            index=0
        )
        
        uploaded_file = None
        if data_source == "Upload CSV File":
            uploaded_file = st.sidebar.file_uploader(
                "Upload sales_data.csv",
                type=['csv'],
                help="CSV should have columns: Date, Product, Region, Sales, Customer_Age, Customer_Gender, Customer_Satisfaction"
            )
            
            if uploaded_file:
                st.sidebar.success("File uploaded successfully!")
        

        st.sidebar.subheader("Display Options")
        show_visualizations = st.sidebar.checkbox("Show Visualizations", value=True)
        show_ai_insights = st.sidebar.checkbox("Show AI Insights", value=True)
        show_detailed_stats = st.sidebar.checkbox("Show Detailed Statistics", value=False)
        
        return data_source, uploaded_file, show_visualizations, show_ai_insights, show_detailed_stats
    
    def load_data(self, data_source: str, uploaded_file) -> Tuple[Any, Any]:
        """Load and process data based on source"""
        processor = DataProcessor()
        
        if data_source == "Upload CSV File" and uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name
            
            try:
                data = processor.load_csv_data(tmp_file_path)
                processed_data = processor.perform_analysis()
                return data, processed_data
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        else:
            data = processor.load_csv_data()
            processed_data = processor.perform_analysis()
            return data, processed_data
    
    def render_data_overview(self, data, processed_data):
        """Render data overview section"""
        st.markdown("## Data Overview")
        

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_records = len(data)
            st.metric("Total Records", f"{total_records:,}")
        
        with col2:
            total_sales = processed_data.get('sales_performance', {}).get('total_sales', 0)
            st.metric("Total Sales", f"${total_sales:,.0f}")
        
        with col3:
            avg_sale = processed_data.get('sales_performance', {}).get('avg_transaction_value', 0)
            st.metric("Avg Sale", f"${avg_sale:,.0f}")
        
        with col4:
            unique_products = len(data['product'].unique()) if 'product' in data.columns else 0
            st.metric("Products", unique_products)
        

        with st.expander("View Sample Data", expanded=False):
            st.dataframe(data.head(10), use_container_width=True)
            

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Date Range:**")
                if 'date' in data.columns:
                    date_range = f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}"
                    st.write(date_range)
                else:
                    st.write("Date information not available")
            
            with col2:
                st.markdown("**Regions:**")
                if 'region' in data.columns:
                    regions = ', '.join(data['region'].unique())
                    st.write(regions)
                else:
                    st.write("Region information not available")
    
    def render_visualizations(self, data, processed_data):
        """Render visualization section"""
        st.markdown("## Data Visualizations")
        
        visualizer = DataVisualizer(data, processed_data)
        

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Sales Trends", 
            "Products", 
            "Regions", 
            "Demographics",
            "Satisfaction"
        ])
        
        with tab1:
            st.markdown("### Sales Performance Over Time")
            try:
                fig = visualizer.create_sales_trend_chart()
                st.plotly_chart(fig, use_container_width=True)
                

                st.markdown("### Daily Patterns")
                fig2 = visualizer.create_temporal_patterns_chart()
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating sales trend charts: {str(e)}")
        
        with tab2:
            st.markdown("### Product Performance Analysis")
            try:
                fig = visualizer.create_product_performance_chart()
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating product chart: {str(e)}")
        
        with tab3:
            st.markdown("### Regional Distribution")
            try:
                fig = visualizer.create_regional_analysis_chart()
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating regional chart: {str(e)}")
        
        with tab4:
            st.markdown("### Customer Demographics")
            try:
                fig = visualizer.create_customer_demographics_chart()
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating demographics chart: {str(e)}")
        
        with tab5:
            st.markdown("### Customer Satisfaction Analysis")
            try:
                fig = visualizer.create_satisfaction_analysis_chart()
                st.plotly_chart(fig, use_container_width=True)
                

                st.markdown("### Variable Correlations")
                fig2 = visualizer.create_correlation_heatmap()
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating satisfaction charts: {str(e)}")
    
    def render_detailed_statistics(self, processed_data):
        """Render detailed statistics section"""
        st.markdown("## Detailed Statistics")
        

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Sales Statistics")
            stats = processed_data.get('statistical_measures', {})
            if stats:
                st.write(f"**Median Sales:** ${stats.get('sales_median', 0):,.2f}")
                st.write(f"**Standard Deviation:** ${stats.get('sales_std', 0):,.2f}")
                st.write(f"**Minimum Sale:** ${stats.get('sales_min', 0):,.2f}")
                st.write(f"**Maximum Sale:** ${stats.get('sales_max', 0):,.2f}")
            
            st.markdown("### Product Performance")
            product_stats = processed_data.get('product_analysis', {})
            if product_stats:
                top_products = product_stats.get('total_sales_by_product', {})
                for product, sales in list(top_products.items())[:3]:
                    st.write(f"**{product}:** ${sales:,.0f}")
        
        with col2:
            st.markdown("### Customer Statistics")
            if stats:
                st.write(f"**Average Age:** {stats.get('avg_customer_age', 0):.1f} years")
                st.write(f"**Average Satisfaction:** {stats.get('avg_satisfaction', 0):.2f}/5.0")
            
            st.markdown("### Regional Performance")
            regional_stats = processed_data.get('regional_analysis', {})
            if regional_stats:
                top_regions = regional_stats.get('total_sales_by_region', {})
                for region, sales in list(top_regions.items())[:3]:
                    st.write(f"**{region}:** ${sales:,.0f}")
    
    def render_ai_insights(self, api_key, processed_data):
        if not AI_AVAILABLE:
            st.info("AI features are disabled (ai_system failed to import).")
            return
        if not api_key:
            st.warning("OpenAI API key is not set in Config or environment.")
            return
        
        st.markdown("## AI-Powered Insights")
        
        # Define predefined questions
        predefined = [
            "What are the key trends in our sales performance?",
            "Where are we over- or under-performing by region?",
            "What should we do next quarter to lift sales?",
        ]
        
        # Initialize RAG system once per session
        if "rag" not in st.session_state:
            with st.spinner("Initializing AI system..."):
                rag = RAGSystem(api_key)
                rag.setup_rag_system(processed_data)
                st.session_state["rag"] = rag
        
        # Initialize chat history
        if "chat" not in st.session_state:
            st.session_state["chat"] = []
        
        rag = st.session_state["rag"]
        
        # Model Evaluation
        with st.expander("Model Evaluation", expanded=False):
            if st.button("Evaluate AI Model", key="btn_eval"):
                try:
                    evaluator = ModelEvaluator(rag)
                    test_qs = predefined[:3]
                    results = evaluator.evaluate_model(test_qs)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Questions", results["total_questions"])
                    with col2:
                        sr = (results["successful_responses"] / results["total_questions"]) * 100 if results["total_questions"] else 0
                        st.metric("Success Rate", f"{sr:.1f}%")
                    
                    for i, r in enumerate(results["results"], start=1):
                        st.markdown(f"**Question {i}**: {r['question']}")
                        txt = r["prediction"]
                        st.write(txt[:500] + "..." if len(txt) > 500 else txt)
                        st.caption(f"Status: {r['evaluation']}")
                        st.markdown("---")
                except Exception as e:
                    st.warning(f"Model evaluation unavailable: {e}")
        
        # Executive Summary
        with st.expander("Executive Summary", expanded=True):
            if st.button("Generate Executive Summary", type="primary", key="btn_summary"):
                with st.spinner("Analyzing your data..."):
                    data_summary = json.dumps(processed_data, indent=2, default=str)[:3000]
                    raw_summary = rag.generate_summary(data_summary)
                    summary = _normalize_llm_text(raw_summary)
                    st.session_state["exec_summary"] = summary
            
            if st.session_state.get("exec_summary"):
                st.markdown(st.session_state["exec_summary"])
        
        # Conversational Chat
        st.markdown("### Chat")
        
        # Question selection
        col1, col2 = st.columns([3, 1])
        with col1:
            picked = st.selectbox("Choose a predefined question:", [""] + predefined, key="q_select")
        with col2:
            if st.button("Use Selected", key="btn_use_selected") and picked:
                st.session_state["pending_user_msg"] = picked
        
        # Show chat history
        for msg in st.session_state["chat"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Handle user input
        pending = st.session_state.get("pending_user_msg")
        user_msg = st.chat_input("Type your question...", key="chat_input")
        
        # If user didn't type but clicked "Use Selected", use that
        if not user_msg and pending:
            user_msg = pending
            if "pending_user_msg" in st.session_state:
                del st.session_state["pending_user_msg"]
        
        if user_msg:
            # Add user message to chat
            st.session_state["chat"].append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Use the correct method name
                    raw_answer = rag.generate_insights(user_msg)
                    answer = _normalize_llm_text(raw_answer)
                    st.markdown(answer)
                    st.session_state["chat"].append({"role": "assistant", "content": answer})
    
    def render_footer(self):
        """Render application footer"""
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                "<div style='text-align: center'>"
                "<p><strong>InsightForge</strong> - Built using LangChain, RAG, and Streamlit</p>"
                "<p>Transform your data into actionable business insights</p>"
                "</div>",
                unsafe_allow_html=True
            )
    
    def run(self):
        """Run the complete Streamlit application"""
        try:

            self.render_header()
            

            data_source, uploaded_file, show_viz, show_ai, show_stats = self.render_sidebar()
            

            with st.spinner("Loading and processing data..."):
                data, processed_data = self.load_data(data_source, uploaded_file)
            

            source_name = "uploaded file" if uploaded_file else "sample data"
            st.success(f"Successfully loaded {len(data):,} records from {source_name}")
            

            self.render_data_overview(data, processed_data)

            api_key = Config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
            
            if show_viz:
                self.render_visualizations(data, processed_data)
            
            if show_stats:
                self.render_detailed_statistics(processed_data)
            
            if show_ai:
                self.render_ai_insights(api_key, processed_data)
            

            self.render_footer()
        
        except Exception as e:
            st.error(f"Application Error: {str(e)}")
            logger.error(f"Application error: {e}")
            

            with st.expander("Debug Information"):
                st.code(str(e))
                st.write("**Suggestions:**")
                st.write("1. Check your CSV file format")
                st.write("2. Verify your OpenAI API key")
                st.write("3. Try using sample data")