#%% 
import streamlit as st
import os

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Mock data for testing (replace with actual retrieval system output)
video_file = "path/to/video.mp4"  # Update with actual video file path or YouTube URL
top_segments = {
    1: {
        'question': 'What is the main topic of the video?',
        'method': 'Cosine Similarity',
        'chunk': 'This is a sample video segment about AI.',
        'distance': 0.85,
        'timestamp': 30
    }
}

def get_rating_emoji(distance):
    """Return an emoji based on cosine distance."""
    return "👍" if distance > 0.8 else "👎"

# Streamlit UI Configuration
st.set_page_config(page_title="Video RAG Retrieval System", layout="wide")

# Title and Description
st.title("Video RAG Retrieval System")
st.markdown("""
This application allows you to query a video and retrieve relevant segments based on your question. 
Enter a question below, and the system will display the most relevant video segment, timestamp, and retrieval details.
If no answer is found, a message will be shown. The video is embedded for easy viewing.
""")

# Layout: Two columns (Video + Chat Interface)
col1, col2 = st.columns([2, 3])

# Column 1: Video Player
with col1:
    st.subheader("Video Content")
    if os.path.exists(video_file):
        st.video(video_file, start_time=0)
    else:
        st.warning(f"Video file {video_file} not found. Please update the path in `app.py`.")
        # Alternative: Embed YouTube video (uncomment if using YouTube URL)
        # st.markdown(f'<iframe width="100%" height="400" src="https://www.youtube.com/embed/dARr31GKwk8" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)

# Column 2: Chat Interface
with col2:
    st.subheader("Ask a Question")
    
    # Chat input
    user_query = st.text_input("Enter your question:", key="user_query")
    
    if user_query:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Placeholder for retrieval logic (integrate with main.py or retrieval system)
        # Simulate retrieval result or no-answer case
        result_found = user_query.lower() in [v["question"].lower() for v in top_segments.values()]
        
        if result_found:
            # Display retrieval result for matching query
            for q_id, segment in top_segments.items():
                if user_query.lower() in segment["question"].lower():
                    response = {
                        "role": "assistant",
                        "content": f"""
**Query ID**: {q_id}  
**Question**: {segment['question']}  
**Top Retrieval Method**: {segment['method']}  
**Top Segment**: {segment['chunk']}  
**Cosine Distance**: {segment['distance']:.3f} ({get_rating_emoji(segment['distance'])})  
**Video Timestamp**: {segment['timestamp']}s  
"""
                    }
                    st.session_state.chat_history.append(response)
                    # Update video start time (if local video)
                    if os.path.exists(video_file):
                        st.video(video_file, start_time=int(segment['timestamp']))
        else:
            # Handle no-answer case
            response = {
                "role": "assistant",
                "content": "Sorry, the answer to your question is not present in the video. Please try another question."
            }
            st.session_state.chat_history.append(response)
    
    # Display chat history
    st.markdown("### Chat History")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You**: {message['content']}")
            else:
                st.markdown(f"**System**: {message['content']}")

# Gold Test Set Results (Optional Display)
st.subheader("Gold Test Set Results")
st.markdown("Below are the results for the predefined gold test set.")
for q_id in sorted(top_segments.keys()):
    with st.expander(f"Query ID {q_id}: {top_segments[q_id]['question']}"):
        st.markdown(f"""
**Top Retrieval Method**: {top_segments[q_id]['method']}  
**Top Segment**: {top_segments[q_id]['chunk']}  
**Cosine Distance**: {top_segments[q_id]['distance']:.3f} ({get_rating_emoji(top_segments[q_id]['distance'])})  
**Video Timestamp**: {top_segments[q_id]['timestamp']}s
""")
        if os.path.exists(video_file):
            st.video(video_file, start_time=int(top_segments[q_id]['timestamp']))
        else:
            st.warning(f"Video file {video_file} not found.")

# %%
