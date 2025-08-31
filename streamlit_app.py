import streamlit as st
import json
import os
from datetime import datetime
from openai import OpenAI
import base64
from pathlib import Path
import tempfile

# Configuration
CHAT_HISTORY_FILE = "chat_history.json"
VECTOR_STORE_CONFIG_FILE = "vector_store_config.json"

def load_chat_history():
    """Load chat history from JSON file"""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def save_chat_history(messages):
    """Save chat history to JSON file"""
    try:
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(messages, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

def load_vector_store_config():
    """Load vector store configuration"""
    if os.path.exists(VECTOR_STORE_CONFIG_FILE):
        try:
            with open(VECTOR_STORE_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_vector_store_config(config):
    """Save vector store configuration"""
    try:
        with open(VECTOR_STORE_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        st.error(f"Error saving vector store config: {e}")

def create_or_get_vector_store(client, store_name="Streamlit Chatbot Knowledge Base"):
    """Create or retrieve existing vector store"""
    config = load_vector_store_config()
    
    # Check if we have an existing vector store
    if 'vector_store_id' in config:
        try:
            # Try to retrieve existing vector store
            vector_store = client.vector_stores.retrieve(config['vector_store_id'])
            return vector_store.id, vector_store.name
        except Exception as e:
            st.warning(f"Previous vector store not found, creating new one: {e}")
    
    # Create new vector store
    try:
        vector_store = client.vector_stores.create(name=store_name)
        config['vector_store_id'] = vector_store.id
        config['vector_store_name'] = vector_store.name
        config['created_at'] = datetime.now().isoformat()
        save_vector_store_config(config)
        return vector_store.id, vector_store.name
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, None

def upload_file_to_vector_store(client, uploaded_file, vector_store_id):
    """Upload a file to OpenAI's vector store"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Upload file to OpenAI
        with open(tmp_file_path, 'rb') as f:
            file_response = client.files.create(
                file=f,
                purpose="assistants"
            )
        
        # Attach file to vector store
        attach_response = client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return {
            "success": True,
            "file_id": file_response.id,
            "vector_store_file_id": attach_response.id,
            "filename": uploaded_file.name
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return {
            "success": False,
            "error": str(e),
            "filename": uploaded_file.name
        }

def process_uploaded_file_for_vision(uploaded_file):
    """Process image files for vision models"""
    if uploaded_file.type.startswith('image/'):
        file_bytes = uploaded_file.read()
        base64_image = base64.b64encode(file_bytes).decode('utf-8')
        return {
            "type": "image",
            "name": uploaded_file.name,
            "data": base64_image,
            "mime_type": uploaded_file.type
        }
    return None

def format_message_with_image(content, image_info):
    """Format message content including image for vision models"""
    if image_info:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_info['mime_type']};base64,{image_info['data']}"
                    }
                }
            ]
        }
    return {"role": "user", "content": content}

def get_vector_store_files(client, vector_store_id):
    """Get list of files in the vector store"""
    try:
        files = client.vector_stores.files.list(vector_store_id=vector_store_id)
        return [{"id": f.id, "created_at": f.created_at} for f in files.data]
    except Exception as e:
        st.error(f"Error retrieving vector store files: {e}")
        return []

# Page configuration
st.set_page_config(
    page_title="💬 AI Chatbot with File Search",
    page_icon="💬",
    layout="wide"
)

# Show title and description
st.title("💬 AI Chatbot with OpenAI File Search")
st.write(
    "This chatbot integrates OpenAI's file search capabilities with persistent chat history. "
    "Upload documents to create a knowledge base that the AI can search through to answer your questions. "
    "Supports text files, PDFs, images, and more!"
)

# Sidebar for settings and file upload
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key input
    openai_api_key = st.text_input("OpenAI API Key", type="password", key="api_key")
    
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        
        # Model selection
        model_choice = st.selectbox(
            "Choose Model",
            ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            index=1  # Default to gpt-4o-mini
        )
        
        st.header("🗂️ Knowledge Base")
        
        # Initialize or get vector store
        if 'vector_store_id' not in st.session_state:
            vector_store_id, vector_store_name = create_or_get_vector_store(client)
            if vector_store_id:
                st.session_state.vector_store_id = vector_store_id
                st.session_state.vector_store_name = vector_store_name
        
        if 'vector_store_id' in st.session_state:
            st.success(f"📁 Knowledge Base: {st.session_state.vector_store_name}")
            st.info(f"ID: {st.session_state.vector_store_id[:20]}...")
            
            # Show files in vector store
            vector_files = get_vector_store_files(client, st.session_state.vector_store_id)
            if vector_files:
                st.write(f"**Files in knowledge base:** {len(vector_files)}")
                with st.expander("View uploaded files"):
                    for i, file_info in enumerate(vector_files[-5:]):  # Show last 5 files
                        st.text(f"File {i+1}: {file_info['created_at']}")
            
            st.header("📁 File Upload")
            uploaded_file = st.file_uploader(
                "Upload to Knowledge Base",
                type=['txt', 'md', 'py', 'js', 'html', 'css', 'json', 'csv', 'pdf', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'webp'],
                help="Upload documents to add to the AI's knowledge base for searching"
            )
            
            if uploaded_file:
                st.info(f"📄 Ready to upload: {uploaded_file.name}")
                st.info(f"File type: {uploaded_file.type}")
                st.info(f"File size: {uploaded_file.size} bytes")
                
                if st.button("Upload to Knowledge Base", type="primary"):
                    with st.spinner("Uploading file to knowledge base..."):
                        result = upload_file_to_vector_store(
                            client, 
                            uploaded_file, 
                            st.session_state.vector_store_id
                        )
                        
                        if result["success"]:
                            st.success(f"✅ Successfully uploaded {result['filename']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to upload {result['filename']}: {result['error']}")
        else:
            st.error("Failed to create/access vector store")
    
    st.header("💾 Chat Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear History", type="secondary"):
            if os.path.exists(CHAT_HISTORY_FILE):
                os.remove(CHAT_HISTORY_FILE)
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Export Chat", type="secondary"):
            if st.session_state.get("messages"):
                chat_data = {
                    "exported_at": datetime.now().isoformat(),
                    "messages": st.session_state.messages,
                    "vector_store_id": st.session_state.get("vector_store_id")
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(chat_data, indent=2),
                    file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Main chat interface
if not openai_api_key:
    st.info("Please add your OpenAI API key in the sidebar to continue.", icon="🗝️")
else:
    # Load existing chat history if not already in session state
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message.get("content"), list):
                # Handle messages with images
                for content_item in message["content"]:
                    if content_item["type"] == "text":
                        st.markdown(content_item["text"])
                    elif content_item["type"] == "image_url":
                        st.image(content_item["image_url"]["url"])
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything! I can search through uploaded documents."):
        
        # Check for uploaded image for vision processing
        uploaded_image_for_vision = None
        if 'uploaded_file' in locals() and uploaded_file and uploaded_file.type.startswith('image/'):
            uploaded_image_for_vision = process_uploaded_file_for_vision(uploaded_file)
        
        # Create message
        if uploaded_image_for_vision:
            message = format_message_with_image(prompt, uploaded_image_for_vision)
            display_content = prompt + f"\n\n📸 Analyzing image: {uploaded_image_for_vision['name']}"
        else:
            message = {"role": "user", "content": prompt}
            display_content = prompt
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(display_content)
            if uploaded_image_for_vision:
                st.image(f"data:{uploaded_image_for_vision['mime_type']};base64,{uploaded_image_for_vision['data']}")
        
        # Add to session state
        st.session_state.messages.append(message)
        
        # Generate AI response
        try:
            # Prepare messages for API call
            api_messages = []
            for msg in st.session_state.messages:
                if isinstance(msg.get("content"), list):
                    # Message with image
                    api_messages.append(msg)
                else:
                    # Regular text message
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Determine if we should use file search
            use_file_search = 'vector_store_id' in st.session_state
            
            with st.chat_message("assistant"):
                if use_file_search:
                    # Use OpenAI Responses API with file search
                    response = client.responses.create(
                        input=prompt,
                        model=model_choice,
                        tools=[{
                            "type": "file_search",
                            "vector_store_ids": [st.session_state.vector_store_id],
                            "max_num_results": 5
                        }],
                        tool_choice="auto"
                    )
                    
                    # Extract the response content
                    if hasattr(response.output[0], 'content'):
                        # File search was used
                        assistant_response = response.output[0].content[0].text
                        
                        # Check if there are annotations (citations)
                        if hasattr(response.output[0].content[0], 'annotations') and response.output[0].content[0].annotations:
                            annotations = response.output[0].content[0].annotations
                            retrieved_files = set([result.filename for result in annotations])
                            
                            # Add citations info
                            if retrieved_files:
                                assistant_response += f"\n\n📚 **Sources consulted:** {', '.join(retrieved_files)}"
                    
                    elif len(response.output) > 1 and hasattr(response.output[1], 'content'):
                        # File search results are in output[1]
                        assistant_response = response.output[1].content[0].text
                        
                        # Check for annotations
                        if hasattr(response.output[1].content[0], 'annotations') and response.output[1].content[0].annotations:
                            annotations = response.output[1].content[0].annotations
                            retrieved_files = set([result.filename for result in annotations])
                            
                            if retrieved_files:
                                assistant_response += f"\n\n📚 **Sources consulted:** {', '.join(retrieved_files)}"
                    else:
                        assistant_response = "I couldn't generate a proper response. Please try again."
                    
                    st.markdown(assistant_response)
                    
                else:
                    # Use regular Chat Completions API
                    stream = client.chat.completions.create(
                        model=model_choice,
                        messages=api_messages,
                        stream=True,
                        max_tokens=2000
                    )
                    assistant_response = st.write_stream(stream)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            # Save updated chat history
            save_chat_history(st.session_state.messages)
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.error("Please check your API key and try again.")
    
    # Display usage information
    if st.session_state.get("messages"):
        st.sidebar.markdown(f"**Total messages:** {len(st.session_state.messages)}")
        
    # Usage instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 How it works")
    st.sidebar.markdown(
        "1. Upload documents to build your knowledge base\n"
        "2. Ask questions - the AI will search through uploaded files\n"
        "3. Get answers with source citations\n"
        "4. Chat history is shared across all users"
    )
    
    if 'vector_store_id' in st.session_state:
        st.sidebar.success("🚀 File search is active!")
    else:
        st.sidebar.info("📝 Using standard chat mode")