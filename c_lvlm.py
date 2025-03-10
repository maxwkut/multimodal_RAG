import os
import lancedb
from embeddings import CLIPEmbeddings
from multimodal_lancedb import MultimodalLanceDB
from openai import OpenAI
import base64
from dotenv import load_dotenv
import gradio as gr
import uuid
from pathlib import Path
from a_preprocessing import download_video, get_transcript_vtt, extract_and_save_frames_and_metadata
from b_create_db import load_and_transform_chunks, store_embeddings
import yt_dlp

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def retrieve_results(query: str, table_name: str, k=5):
    """Retrieve results based on query from vectorstore."""
    db = lancedb.connect(".lancedb")

    embedder = CLIPEmbeddings()
    # Creating a LanceDB vector store
    vectorstore = MultimodalLanceDB(
        uri=".lancedb", embedding=embedder, table_name=table_name
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    results = retriever.invoke(query)
    return results

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_content_array(results, user_query):
    formatted_context = "\n\n".join([f"Document {i+1}: {doc.metadata['transcript']}" for i, doc in enumerate(results)])
    
    content_array = [
        {"type": "text", "text": f"Retrieved information: {formatted_context}\n\nMy question: {user_query}"},
    ]

    # Add each JPG image to the content array
    for i, doc in enumerate(results):
        # Encode the image
        base64_image = encode_image_to_base64(doc.metadata['extracted_frame_path'])
        
        # Add to content array with JPEG mime type
        content_array.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    
    return content_array, results

# Function to handle multi-turn conversations
def chat_with_llm(messages, model="gpt-4o-mini"):
    """Handle multi-turn conversations with the LLM."""
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return chat_completion.choices[0].message

# Initialize conversation with system message
def initialize_conversation():
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant analyzing an unploaded video that is split into transcript chunks and frames. Use the retrieved frames and transcripts to answer the user's question in a clear and concise way. If the user asks about something not in the current context, indicate that you need to search for more information."
        }
    ]

# Function to determine if a new retrieval is needed
def needs_retrieval(user_input, conversation_history):
    # Create a message to ask the LLM if retrieval is needed
    retrieval_check_messages = [
        {
            "role": "system",
            "content": "You determine if a user's question requires retrieving information from a video database. Respond with 'YES' if the question is about specific video content that requires retrieval. Respond with 'NO' if the question is general conversation (like greetings or how are you), can be answered with general knowledge, or can be answered using the existing conversation context without new retrieval. Answer only YES or NO."
        }
    ]
    
    # Add the conversation history for context
    for message in conversation_history[1:]:  # Skip system message
        retrieval_check_messages.append(message)
    
    # Add the new question
    retrieval_check_messages.append({
        "role": "user",
        "content": f"Does this question require retrieving information from the video database? '{user_input}'"
    })
    
    # Get decision from LLM
    decision = chat_with_llm(retrieval_check_messages)
    return "YES" in decision.content.upper()

# Function to add a new user message and get response
def send_message(user_input, conversation_history, table_name="test_tbl"):
    # Check if we need to retrieve new information
    retrieved_content = None
    if needs_retrieval(user_input, conversation_history):
        # Get new retrieval results
        results = retrieve_results(query=user_input, table_name=table_name, k=5)
        content_array, retrieved_content = prepare_content_array(results, user_input)
        conversation_history.append({
            "role": "user",
            "content": content_array
        })
    else:
        conversation_history.append({
            "role": "user", 
            "content": user_input
        })
    
    # Get response from LLM
    response = chat_with_llm(conversation_history)
    
    # Add the response to history
    conversation_history.append({
        "role": "assistant",
        "content": response.content
    })
    
    return response.content, conversation_history, retrieved_content

# Function to add a new user message and get response
def process_initial_query(user_query, table_name="test_tbl"):
    conversation_history = initialize_conversation()
    retrieved_content = None
    
    # For the initial query, determine if it requires video content retrieval
    retrieval_check_messages = [
        {
            "role": "system",
            "content": "You determine if a user's question requires retrieving information from a video database. Respond with 'YES' if the question is about specific video content that requires retrieval. Respond with 'NO' if the question is general conversation (like greetings or how are you) or can be answered with general knowledge. Answer only YES or NO."
        },
        {
            "role": "user",
            "content": f"Does this question require retrieving information from the video database? '{user_query}'"
        }
    ]
    
    decision = chat_with_llm(retrieval_check_messages)
    needs_video_retrieval = "YES" in decision.content.upper()
    
    if needs_video_retrieval:
        results = retrieve_results(query=user_query, table_name=table_name, k=5)
        content_array, retrieved_content = prepare_content_array(results, user_query)
        conversation_history.append({
            "role": "user",
            "content": content_array
        })
    else:
        conversation_history.append({
            "role": "user", 
            "content": user_query
        })
    
    # Get initial response
    response = chat_with_llm(conversation_history)
    
    # Add the response to conversation history
    conversation_history.append({
        "role": "assistant",
        "content": response.content
    })
    
    return response.content, conversation_history, retrieved_content

def format_retrieved_content(retrieved_docs):
    if not retrieved_docs:
        return "<div style='text-align:center; padding:50px 20px; color:#8e8ea0;'>No content retrieved yet</div>"
    
    html_content = "<div style='overflow-y: auto; max-height: 500px;'>"
    
    for i, doc in enumerate(retrieved_docs):
        # Add transcript
        html_content += f"<h3 style='color:#ececec; font-weight:500; margin-bottom:8px;'>Document {i+1}</h3>"
        html_content += f"<p><strong style='color:#8e8ea0;'>Transcript:</strong> <span style='color:#ececec;'>{doc.metadata['transcript']}</span></p>"
        
        # Add image - using base64 encoding for proper rendering
        image_path = doc.metadata['extracted_frame_path']
        base64_image = encode_image_to_base64(image_path)
        html_content += f"<img src='data:image/jpeg;base64,{base64_image}' style='max-width: 100%; margin-bottom: 20px; border-radius: 6px; border: 1px solid #333333;'>"
    
    html_content += "</div>"
    return html_content

def gradio_chat(message, history, conversation_state=None, retrieved_content_html=None, table_name="test_tbl"):
    if conversation_state is None:
        # First message - initialize and process with retrieval
        bot_message, conversation_state, retrieved_docs = process_initial_query(message, table_name)
    else:
        # Follow-up message - continue the conversation
        bot_message, conversation_state, retrieved_docs = send_message(message, conversation_state, table_name)
    
    # Format retrieved content if available
    if retrieved_docs:
        retrieved_content_html = format_retrieved_content(retrieved_docs)
    
    # Return in the format expected by Gradio chatbot (list of message pairs)
    return history + [[message, bot_message]], conversation_state, retrieved_content_html

def process_youtube_video(youtube_url):
    # Create a unique session ID for this video
    session_id = str(uuid.uuid4())[:8]
    table_name = f"video_{session_id}"
    
    # Create directories for this session
    base_dir = f"data/videos/{session_id}"
    extracted_frames_path = os.path.join(base_dir, "extracted_frame")
    
    # Create these output folders if not existing
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)

    # Get video info to extract title
    with yt_dlp.YoutubeDL() as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_title = info_dict.get('title', 'Unknown Video')
    
    video_filepath = download_video(youtube_url, base_dir)
    transcript_filepath = get_transcript_vtt(youtube_url, base_dir)

    # Preprocess
    _ = extract_and_save_frames_and_metadata(
        video_filepath,
        transcript_filepath,
        extracted_frames_path,
        base_dir,
    )

    metadata_path = os.path.join(base_dir, "metadata.json")

    # Create vector database
    metadata, transcripts, frame_paths = load_and_transform_chunks(metadata_path)
    
    # Store embeddings
    store_embeddings(
        transcripts=transcripts,
        frame_paths=frame_paths,
        metadata=metadata,
        table_name=table_name
    )
    return table_name, video_title

if __name__ == "__main__":
    # Load CSS from external file
    with open("style.css", "r") as f:
        custom_css = f.read()

    with gr.Blocks(title="VideoInsight AI", css=custom_css) as demo:
        # demo.load()
        
        with gr.Column(elem_classes="container"):
            # Header
            with gr.Column(elem_classes="main-header"):
                gr.Markdown("# VideoInsight AI", elem_classes="main-title")
                gr.Markdown("Intelligent Video Analysis & Conversation System", elem_classes="subtitle")
            
            # Video Source Section - Full Width
            with gr.Column(elem_classes="card upload-card") as upload_section:
                gr.Markdown("## Video Source", elem_classes="section-header")
                
                with gr.Column() as upload_form:
                    with gr.Row():
                        with gr.Column(scale=4):
                            youtube_url = gr.Textbox(
                                label="YouTube URL", 
                                placeholder="Enter a YouTube video URL",
                                elem_classes="input-box"
                            )
                        with gr.Column(scale=1):
                            process_btn = gr.Button("Process Video", elem_classes="process-btn")
            
            # Chat and Retrieved Content Sections - Two Columns
            with gr.Row():
                # Left column - Chat
                with gr.Column(scale=1):
                    # Chat section
                    with gr.Column(elem_classes="card animate-fade-in chat-section"):
                        gr.Markdown("## Chat with Video", elem_classes="section-header")
                        chatbot = gr.Chatbot(height=500, elem_classes="chatbot-container")
                        with gr.Row():
                            msg = gr.Textbox(
                                label="Message", 
                                placeholder="Ask a question about the video content...",
                                elem_classes="input-box"
                            )
                            clear = gr.Button("Clear", elem_classes="clear-btn")
                
                # Right column - Retrieved content
                with gr.Column(scale=1, elem_classes="retrieved-section"):
                    with gr.Column(elem_classes="card animate-fade-in"):
                        gr.Markdown("## Retrieved Content", elem_classes="section-header")
                        retrieved_content_html = gr.HTML(
                            value="<div style='text-align:center; padding:50px 20px; color:#8e8ea0;'>No content retrieved yet. Process a video and ask a question to see retrieved frames and transcripts here.</div>",
                            elem_classes="retrieved-content"
                        )
            
            # Footer
            gr.Markdown(
                "© 2025 Max Kutschinski", 
                elem_classes="footer"
            )
        
        # State variables
        table_name = gr.State(value=None)  # Table name for the processed video
        conversation_state = gr.State()  # Conversation history
        title_markdown = gr.Markdown()

        # Process video function
        def on_process(url):
            new_table_name, video_title = process_youtube_video(url)
            
            title_markdown = f"**{video_title}**"
            
            return new_table_name, title_markdown

        
        # Connect the process button
        process_btn.click(
            fn=on_process,
            inputs=[youtube_url],
            outputs=[table_name, title_markdown]
        )

        # Define the submit event handler
        def on_submit(message, history, conversation_state, retrieved_content_html, table_name):
            if message.strip() == "":
                return "", history, conversation_state, retrieved_content_html
            
            if table_name is None or table_name == "":
                return "", history + [["", "Please process a YouTube video first before chatting."]], conversation_state, retrieved_content_html
            
            new_history, new_state, new_retrieved_content = gradio_chat(
                message, history or [], conversation_state, retrieved_content_html, table_name
            )
            return "", new_history, new_state, new_retrieved_content or retrieved_content_html
        
        # Connect the submit event
        msg.submit(
            fn=on_submit,
            inputs=[msg, chatbot, conversation_state, retrieved_content_html, table_name],
            outputs=[msg, chatbot, conversation_state, retrieved_content_html]
        )
        
        # Define the clear function
        def on_clear():
            return "", [], None, "<div style='text-align:center; padding:50px 20px; color:#8e8ea0;'>No content retrieved yet. Process a video and ask a question to see retrieved frames and transcripts here.</div>"
        
        # Connect the clear button
        clear.click(
            fn=on_clear,
            outputs=[msg, chatbot, conversation_state, retrieved_content_html]
        )
    
    demo.launch()