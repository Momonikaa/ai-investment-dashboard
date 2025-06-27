import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from backend.pdf_utils import extract_text_from_pdf, chunk_text
from backend.embedding_utils import get_embeddings, build_faiss_index
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.express as px

st.title("AI-Powered Investment Research Dashboard")

# --- LLM response functions ---

def generate_local_response(prompt, context):
    """Generate a basic response using the retrieved context (no real LLM)"""
    if not context.strip():
        return "I couldn't find relevant information in your uploaded documents. Please try rephrasing your question."
    return f"""Based on your research documents, here's what I found:

**Key Information:**
{context[:1000]}...

**Summary:** This appears to be related to your question about "{prompt}". The documents contain relevant information that can help answer your query.

*Note: This is a basic response. For more sophisticated analysis, please use a paid LLM option with your API key.*"""

def generate_openai_response(prompt, context, api_key, model_choice):
    """Generate response using OpenAI API"""
    try:
        import openai
        openai.api_key = api_key
        model_name = "gpt-4" if "GPT-4" in model_choice else "gpt-3.5-turbo"
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a financial research assistant. Answer questions based on the provided research documents context."},
                {"role": "user", "content": f"Context from research documents:\n{context}\n\nQuestion: {prompt}"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"

def generate_claude_response(prompt, context, api_key):
    """Generate response using Anthropic Claude API"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        model = "claude-3-5-sonnet-20240620"  # Use the latest Claude 3.5 Sonnet model
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": f"Context from research documents:\n{context}\n\nQuestion: {prompt}"}
            ],
        )
        # Claude returns a list of content blocks; join text blocks
        return "".join([block.text for block in message.content if hasattr(block, "text")])
    except Exception as e:
        return f"Error calling Claude API: {str(e)}"

def generate_gemini_response(prompt, context, api_key):
    """Generate response using Google Gemini API"""
    try:
        from google import genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        full_prompt = f"Context from research documents:\n{context}\n\nQuestion: {prompt}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

# --- Sidebar for LLM configuration ---
st.sidebar.header("LLM Configuration")
llm_choice = st.sidebar.selectbox(
    "Choose LLM Model",
    ["Local (Free)", "OpenAI GPT-4", "OpenAI GPT-3.5", "Anthropic Claude", "Google Gemini"]
)

api_key = ""
if llm_choice != "Local (Free)":
    api_key = st.sidebar.text_input("Enter API Key", type="password")
    if not api_key:
        st.sidebar.warning("Please enter your API key to use this model")

# --- File upload section ---
uploaded_files = st.file_uploader(
    "Upload one or more PDF research reports", type="pdf", accept_multiple_files=True
)

# Create directory to save uploaded files
save_dir = os.path.join(project_root, "assets", "uploads")
os.makedirs(save_dir, exist_ok=True)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s)")
    
    # Progress bar for processing
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_texts, all_chunks = [], []
    
    for i, pdf in enumerate(uploaded_files):
        # Save uploaded file to disk
        file_path = os.path.join(save_dir, pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())
        
        status_text.text(f"Processing {pdf.name}...")
        
        # Extract text and chunk
        text = extract_text_from_pdf(pdf)
        all_texts.append(text)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
        
    status_text.text("Generating embeddings...")
    embeddings = get_embeddings(all_chunks)
    index = build_faiss_index(embeddings)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    st.success(f"Processed {len(all_chunks)} text chunks from {len(uploaded_files)} PDFs")
    st.info(f"Files saved to: {save_dir}")
    
    # --- Display file details ---
    with st.expander("View uploaded files"):
        for i, pdf in enumerate(uploaded_files):
            st.write(f"📄 **{pdf.name}** - {len(all_texts[i]):,} characters")

    # --- Portfolio Builder Section ---
    stock_data_path = os.path.join(project_root, "assets", "merged_stock_data.xlsx")
    stock_df = pd.read_excel(stock_data_path)
    stock_df.columns = stock_df.columns.str.strip()

    st.header("📈 Build Your Stock Portfolio")

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

    # Input for adding a stock
    with st.form("add_stock_form", clear_on_submit=True):
        symbol = st.text_input("Enter Stock Symbol (e.g., TCS, INFY):").upper()
        shares = st.number_input("Number of Shares", min_value=1, value=1)
        add = st.form_submit_button("Add to Portfolio")

        if add and symbol:
            if symbol in stock_df['Stock Symbol'].values:
                st.session_state.portfolio.append({"Stock Symbol": symbol, "Shares": shares})
                st.success(f"Added {shares} shares of {symbol} to your portfolio.")
            else:
                st.error(f"Stock Symbol '{symbol}' not found in data.")

    # Show current portfolio
    if st.session_state.portfolio:
        st.subheader("Your Portfolio")
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        st.dataframe(portfolio_df)
    else:
        st.info("Add stocks to build your portfolio.")

    if st.session_state.portfolio:
        # Merge user portfolio with stock data
        merged = pd.merge(pd.DataFrame(st.session_state.portfolio), stock_df, on="Stock Symbol", how="left")

        # Show key metrics
        metrics_cols = [
            "Stock Symbol", "Company Name", "Shares", "Sector", "Volatility", "Beta", "P/E", "Max Drawdown", "VaR (95%)", 
            "RSI", "MACD", "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio"
        ]
        available_cols = [col for col in metrics_cols if col in merged.columns]
        st.subheader("Portfolio Risk & Performance Metrics")
        st.dataframe(merged[available_cols])
    
        # Example: Pie chart of sector allocation
        if "Sector" in merged.columns:
            sector_counts = merged["Sector"].value_counts().reset_index()
            sector_counts.columns = ["Sector", "Count"]
            fig = px.pie(sector_counts, names="Sector", values="Count", title="Sector Allocation")
            st.plotly_chart(fig)

        # Example: Bar chart of volatility
        if "Volatility" in merged.columns:
            fig2 = px.bar(merged, x="Stock Symbol", y="Volatility", title="Stock Volatility")
            st.plotly_chart(fig2)

    # --- Chat interface ---
    st.subheader("💬 Ask Questions About Your Research Reports")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your research reports..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Retrieve relevant chunks
        with st.spinner("Searching through your documents..."):
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_embedding = model.encode([prompt])
            D, I = index.search(np.array(query_embedding), 5)
            context = "\n".join([all_chunks[i] for i in I[0] if i < len(all_chunks)])

        # Generate response based on selected LLM
        with st.spinner("Generating response..."):
            if llm_choice == "Local (Free)":
                response = generate_local_response(prompt, context)
            elif llm_choice.startswith("OpenAI") and api_key:
                response = generate_openai_response(prompt, context, api_key, llm_choice)
            elif llm_choice == "Anthropic Claude" and api_key:
                response = generate_claude_response(prompt, context, api_key)
            elif llm_choice == "Google Gemini" and api_key:
                response = generate_gemini_response(prompt, context, api_key)
            else:
                response = "⚠️ Please select a model and provide API key if required."

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Show sources
        with st.expander("📚 View Sources"):
            st.write("**Relevant text chunks used:**")
            for i, chunk_idx in enumerate(I[0][:3]):
                if chunk_idx < len(all_chunks):
                    st.write(f"**Source {i+1}:**")
                    st.write(all_chunks[chunk_idx][:300] + "...")
                    st.write("---")

# Footer
st.markdown("---")
st.markdown("🚀 **AI-Powered Investment Research Dashboard** | Upload PDFs → Ask Questions → Get Insights")
