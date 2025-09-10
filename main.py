import streamlit as st
import os
import logging
from ui_components import StreamlitApp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main application entry point"""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()