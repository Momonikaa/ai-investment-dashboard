import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer
from backend.pdf_utils import extract_text_from_pdf, chunk_text
from backend.embedding_utils import get_embeddings, build_faiss_index

# Initialize project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- LLM Response Functions ---
def generate_local_response(prompt, context, portfolio_summary=""):
    """Generate response using local/free method"""
    if not context.strip():
        return "No relevant information found. Try rephrasing your question."
    
    base_response = f"""
**Research Context:**
{context[:1000]}...

**Summary:** This relates to your question about "{prompt}".
"""
    if portfolio_summary:
        base_response += f"\n**Portfolio Context:**\n{portfolio_summary}"
    return base_response

def generate_openai_response(prompt, context, api_key, model_choice, portfolio_summary=""):
    """Generate response using OpenAI API"""
    try:
        import openai
        openai.api_key = api_key
        
        model_name = "gpt-4" if "GPT-4" in model_choice else "gpt-3.5-turbo"
        
        messages = [
            {"role": "system", "content": "You're a financial research assistant. Use both research context and portfolio data."},
            {"role": "user", "content": f"Research Context:\n{context}\n\nPortfolio Data:\n{portfolio_summary}\n\nQuestion: {prompt}"}
        ]
        
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    except Exception as e:
        return f"OpenAI Error: {str(e)}"

def generate_claude_response(prompt, context, api_key, portfolio_summary=""):
    """Generate response using Anthropic Claude API"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt_text = f"""Research Context:
{context}

Portfolio Data:
{portfolio_summary}

Question: {prompt}"""
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt_text}]
        )
        return "".join(block.text for block in message.content)
    
    except Exception as e:
        return f"Claude Error: {str(e)}"

def generate_gemini_response(prompt, context, api_key, portfolio_summary=""):
    """Generate response using Google Gemini API"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        full_prompt = f"""
Research Context:
{context}

Portfolio Data:
{portfolio_summary}

Question: {prompt}"""
        
        response = model.generate_content(full_prompt)
        return response.text
    
    except Exception as e:
        return f"Gemini Error: {str(e)}"

# --- Risk Calculation Functions ---
def calculate_risk_scores(merged_df):
    """Calculate risk scores for portfolio"""
    risk_data = []
    
    for _, row in merged_df.iterrows():
        risk_entry = {"Stock Symbol": row["Stock Symbol"]}
        
        # Valuation Risk (P/E)
        pe = row.get("P/E", 0)
        risk_entry["Valuation Risk"] = "🟢" if pe < 20 else "🟡" if pe < 30 else "🔴"
        
        # Market Risk (Beta)
        beta = row.get("Beta", 1)
        risk_entry["Market Risk"] = "🟢" if beta < 0.9 else "🟡" if beta < 1.1 else "🔴"
        
        # Liquidity Risk (Volume)
        volume = row.get("Volume", 0)
        risk_entry["Liquidity Risk"] = "🔴" if volume < 100000 else "🟡" if volume < 500000 else "🟢"
        
        # Credit Risk (Debt-to-Equity)
        de_ratio = row.get("Debt-to-Equity", 0)
        risk_entry["Credit Risk"] = "🟢" if de_ratio < 0.5 else "🟡" if de_ratio < 1.0 else "🔴"
        
        risk_data.append(risk_entry)
    
    return pd.DataFrame(risk_data)

def monte_carlo_simulation(portfolio, simulations=1000, days=252):
    """Run Monte Carlo simulation for portfolio"""
    simulated_returns = []
    
    for _ in range(simulations):
        portfolio_return = 0
        for _, stock in portfolio.iterrows():
            daily_vol = stock["Volatility"] / np.sqrt(252)
            daily_return = np.random.normal(0, daily_vol)
            portfolio_return += daily_return * stock["Weight"]
        simulated_returns.append(portfolio_return)
    
    return pd.DataFrame(simulated_returns, columns=["Daily Return"])

# --- Main App ---
st.title("AI-Powered Investment Research Dashboard")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
llm_choice = st.sidebar.selectbox(
    "LLM Model",
    ["Local (Free)", "OpenAI GPT-4", "OpenAI GPT-3.5", "Anthropic Claude", "Google Gemini"]
)

api_key = ""
if llm_choice != "Local (Free)":
    api_key = st.sidebar.text_input("API Key", type="password")

# --- File Processing Section ---
uploaded_files = st.file_uploader(
    "Upload PDF Research Reports", 
    type="pdf", 
    accept_multiple_files=True
)

save_dir = os.path.join(project_root, "assets", "uploads")
os.makedirs(save_dir, exist_ok=True)

if uploaded_files:
    st.success(f"Processing {len(uploaded_files)} file(s)...")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    all_texts, all_chunks = [], []
    
    for i, pdf in enumerate(uploaded_files):
        # Save file
        file_path = os.path.join(save_dir, pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())
        
        # Process text
        status_text.text(f"Extracting: {pdf.name}")
        text = extract_text_from_pdf(pdf)
        all_texts.append(text)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    # Generate embeddings
    status_text.text("Generating embeddings...")
    embeddings = get_embeddings(all_chunks)
    index = build_faiss_index(embeddings)
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    st.info(f"Processed {len(all_chunks)} text chunks")

    # File details expander
    with st.expander("Uploaded Files"):
        for i, pdf in enumerate(uploaded_files):
            st.write(f"📄 {pdf.name} - {len(all_texts[i]):,} chars")

    # --- Portfolio Builder Section ---
    st.header("📈 Portfolio Management")
    stock_data_path = os.path.join(project_root, "assets", "merged_stock_data.xlsx")
    stock_df = pd.read_excel(stock_data_path)
    stock_df.columns = stock_df.columns.str.strip()

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

    # Add stocks form
    with st.form("add_stock_form", clear_on_submit=True):
        symbol = st.text_input("Stock Symbol (e.g., TCS, INFY):").upper()
        shares = st.number_input("Shares", min_value=1, value=1)
        add = st.form_submit_button("Add to Portfolio")
        
        if add and symbol:
            if symbol in stock_df['Stock Symbol'].values:
                st.session_state.portfolio.append({
                    "Stock Symbol": symbol, 
                    "Shares": shares
                })
                st.success(f"Added {shares} shares of {symbol}")
            else:
                st.error(f"Symbol '{symbol}' not found")

    # Portfolio display
    if st.session_state.portfolio:
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        st.subheader("Your Portfolio")
        st.dataframe(portfolio_df)
        
        # Merge with stock data
        merged = pd.merge(
            portfolio_df, 
            stock_df, 
            on="Stock Symbol", 
            how="left"
        )
        st.session_state.merged = merged  # Store for later use
        
        # Calculate weights
        total_value = (merged["Shares"] * merged["Price"]).sum()
        merged["Weight"] = (merged["Shares"] * merged["Price"]) / total_value
        
        # Display metrics
        metrics_cols = [
            "Stock Symbol", "Company Name", "Shares", "Sector", 
            "Price", "Weight", "Volatility", "Beta", "P/E", 
            "Max Drawdown", "VaR (95%)", "RSI", "MACD", 
            "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio"
        ]
        available_cols = [col for col in metrics_cols if col in merged.columns]
        st.dataframe(merged[available_cols])
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            if "Sector" in merged.columns:
                fig = px.pie(
                    merged, 
                    names="Sector", 
                    values="Weight",
                    title="Sector Allocation"
                )
                st.plotly_chart(fig)
        
        with col2:
            if "Volatility" in merged.columns:
                fig = px.bar(
                    merged, 
                    x="Stock Symbol", 
                    y="Volatility",
                    title="Stock Volatility"
                )
                st.plotly_chart(fig)
        
        # --- Risk Analytics Section ---
        st.subheader("🔍 Risk Analytics")
        risk_df = calculate_risk_scores(merged)
        st.dataframe(risk_df)
        
        # Risk summary
        st.markdown("**Risk Legend:** 🟢 Low 🟡 Medium 🔴 High")
        
        # --- Stress Testing ---
        st.subheader("🧪 Stress Testing")
        if st.button("Run Monte Carlo Simulation"):
            with st.spinner("Running 1000 simulations..."):
                results = monte_carlo_simulation(merged)
                fig = px.histogram(
                    results, 
                    x="Daily Return",
                    title="Simulated Daily Returns Distribution",
                    nbins=50
                )
                st.plotly_chart(fig)
                st.caption("Based on historical volatility and random walk assumption")
    
    else:
        st.info("Add stocks to build your portfolio")

    # --- Research Q&A Section ---
    st.header("💬 Research Analysis")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about research reports..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Retrieve relevant context
        with st.spinner("Searching documents..."):
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_embedding = model.encode([prompt])
            D, I = index.search(np.array(query_embedding), 5)
            context = "\n".join([
                all_chunks[i] 
                for i in I[0] 
                if i < len(all_chunks)
            ])
        
        # Prepare portfolio context
        portfolio_summary = ""
        if "merged" in st.session_state and not st.session_state.merged.empty:
            portfolio_summary = st.session_state.merged[
                ["Stock Symbol", "Shares", "Sector", "Weight"]
            ].to_markdown(index=False)
        
        # Generate response
        with st.spinner("Generating insights..."):
            if llm_choice == "Local (Free)":
                response = generate_local_response(prompt, context, portfolio_summary)
            elif llm_choice.startswith("OpenAI") and api_key:
                response = generate_openai_response(prompt, context, api_key, llm_choice, portfolio_summary)
            elif llm_choice == "Anthropic Claude" and api_key:
                response = generate_claude_response(prompt, context, api_key, portfolio_summary)
            elif llm_choice == "Google Gemini" and api_key:
                response = generate_gemini_response(prompt, context, api_key, portfolio_summary)
            else:
                response = "⚠️ Select model and provide API key if required"
        
        # Display response
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Show sources
        with st.expander("📚 Sources Used"):
            for i, idx in enumerate(I[0][:3]):
                if idx < len(all_chunks):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(all_chunks[idx][:300] + " [...]")

# --- Footer ---
st.markdown("---")
st.markdown("🚀 **AI Investment Dashboard** • Upload PDFs • Analyze Portfolios • Get Insights")
