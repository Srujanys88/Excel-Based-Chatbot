import streamlit as st
import pandas as pd
import os
import re
import plotly.express as px
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Set your OPENAI_API_KEY")
    st.stop()

client = OpenAI()
st.set_page_config(page_title="Excel Chatbot", layout="wide")
st.title("Natural Language Chabot for Excel-Based Insights ")

def normalize_columns(df):
    new_cols = {}
    for col in df.columns:
        clean_col = re.sub(r'[^A-Za-z0-9_]', '_', col.strip().lower())
        new_cols[col] = clean_col
    return df.rename(columns=new_cols)

def get_column_info(df):
    info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isnull().sum()
        unique = df[col].nunique(dropna=True)
        info.append(f"{col}: type={dtype}, nulls={nulls}, unique={unique}")
    return "\n".join(info)

def process_query(question, df):
    sample = df.head(10).to_csv(index=False)
    schema = get_column_info(df)
    prompt = f"""
You are a helpful data assistant.

Here is a sample of the dataset:
{sample}

Schema:
{schema}

Answer the following question in plain English. Do not include any Python code:
{question}
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

def extract_plot_columns(question, df):
    prompt = f"""
You are a visualization assistant.
Columns in the dataset: {list(df.columns)}
User query: "{question}"
Return a JSON like:
{{"x": "column1", "y": "column2"}} or {{"x": "column1"}}
Only include keys that apply.
"""
    try:
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {}

def try_visualization(df, question):
    keywords = ["plot", "chart", "graph", "visual", "histogram", "bar", "scatter", "pie"]
    if not any(word in question.lower() for word in keywords):
        return None
    cols = extract_plot_columns(question, df)
    try:
        if "x" in cols and "y" in cols:
            return px.bar(df, x=cols["x"], y=cols["y"])
        elif "x" in cols:
            if pd.api.types.is_numeric_dtype(df[cols["x"]]):
                return px.histogram(df, x=cols["x"])
            else:
                return px.bar(df[cols["x"]].value_counts().reset_index(), x='index', y=cols["x"])
    except:
        return None
    return None

def chat_interface():
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df = normalize_columns(df)
        st.session_state.df = df
        st.success("File uploaded and processed")
        st.dataframe(df.head(), use_container_width=True)
    if 'df' not in st.session_state:
        st.warning("Please upload an Excel file to get started.")
        return
    df = st.session_state.df
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask your question about the data"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    answer = process_query(prompt, df)
                    fig = try_visualization(df, prompt)
                    st.markdown(answer)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f" Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    if st.button("Clear Chat"):
        st.session_state.messages = []

if __name__ == '__main__':
    chat_interface()