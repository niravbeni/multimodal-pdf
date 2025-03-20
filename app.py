def initialize_session_state():
    """Initialize session state variables"""
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    else:
        # Add formatted_content field to existing messages if needed
        for message in st.session_state.conversation:
            if "formatted_content" not in message:
                message["formatted_content"] = message["content"]
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "mode" not in st.session_state:
        st.session_state.mode = "upload"  # Default to upload mode
    
    if "temp_pdf_files" not in st.session_state:
        st.session_state.temp_pdf_files = []

def display_conversation():
    """Display the conversation history using Streamlit's chat interface"""
    # Get messages
    messages = st.session_state.conversation
    if not messages:  # If no messages, return early
        return

    # Create a container for the messages
    with st.container():
        # Show full history
        for message in messages:
            avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
            with st.chat_message(message["role"], avatar=avatar):
                # Use formatted content if available, otherwise use plain content
                content = message.get("formatted_content", message["content"])
                st.markdown(content, unsafe_allow_html=True)