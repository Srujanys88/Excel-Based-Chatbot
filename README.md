# Excel-Based Chatbot for Natural Language Insights

This project is a **Streamlit-based web application** that enables users to interact with **Excel files using natural language**. Powered by **OpenAI's GPT-3.5 model**, the chatbot can interpret user queries and return insights from the Excel data, along with visualizations when appropriate.

---

## 🚀 Features

- Upload `.xlsx` or `.xls` Excel files
- Normalize messy column names for consistent querying
- Ask natural language questions about your dataset
- Generate data insights in plain English
- Auto-generate visualizations (bar charts, histograms) based on the query
- Persistent conversation history during the session
- Clear chat history with one click

---

## 🧠 Tech Stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [OpenAI GPT-3.5 Turbo](https://platform.openai.com/)
- [Plotly Express](https://plotly.com/python/plotly-express/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

---

## 📁 Project Structure

.
├── app.py # Main application script
├── .env # Environment file for sensitive keys (DO NOT COMMIT)
├── requirements.txt # Python package dependencies
└── README.md # Project documentation