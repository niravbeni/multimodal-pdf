"""
HTML and CSS styling for Text PDF Chat
"""

def inject_css():
    """Return CSS for styling the chat interface"""
    return """
    <style>
    /* Main container styling */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        padding: 1rem 0;
    }
    
    /* User message styling */
    [data-testid="stChatMessage"][data-chat-message-user-name="user"] {
        background-color: white;
    }
    
    /* Assistant message styling */
    [data-testid="stChatMessage"][data-chat-message-user-name="assistant"] {
        background-color: #f7f7f8;
    }
    
    /* Add padding to chat message container */
    .stChatMessage {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        margin-bottom: 1rem;
    }
    
    /* Style code blocks */
    pre {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        overflow-x: auto;
    }
    
    /* Style info messages */
    .info-box {
        background-color: #e8f0fe;
        border-left: 5px solid #4285f4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    
    /* Style error messages */
    .error-box {
        background-color: #fce8e8;
        border-left: 5px solid #ea4335;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    
    /* Citation styling */
    .citation, code {
        display: inline-block;
        margin: 0 2px;
        margin-top: 12px;
        padding: 4px 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        font-weight: 500;
        background-color: #f0f2f6;
        color: #0066cc;
        border: 1px solid #cfd8dc;
        border-radius: 4px;
        white-space: nowrap;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    </style>
    """