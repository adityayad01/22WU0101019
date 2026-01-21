import streamlit as st
import sys
sys.path.append('..')

from llm.model_handler import LLaMAHandler
from vector_db.pinecone_handler import PineconeHandler

class ChatbotInterface:
    def __init__(self):
        self.llm_handler = LLaMAHandler()
        self.pinecone_handler = PineconeHandler()
        
    def initialize(self):
        if 'initialized' not in st.session_state:
            with st.spinner("Loading Jarvis..."):
                self.llm_handler.load_model()
                self.pinecone_handler.initialize()
                st.session_state.initialized = True
                st.session_state.messages = []
    
    def run(self):
        st.set_page_config(page_title="Jarvis AI Assistant", page_icon="🤖")
        st.title("🤖 Jarvis - Your AI Assistant")
        
        self.initialize()
        
        # Sidebar for knowledge management
        with st.sidebar:
            st.header("Knowledge Base")
            new_knowledge = st.text_area("Add new knowledge:")
            if st.button("Add to Knowledge Base"):
                if new_knowledge:
                    self.pinecone_handler.add_knowledge(new_knowledge)
                    st.success("Knowledge added!")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask Jarvis anything..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get context and generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    context = self.pinecone_handler.search_knowledge(prompt)
                    full_prompt = self.llm_handler.create_chat_prompt(prompt, context)
                    response = self.llm_handler.generate_response(full_prompt)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})