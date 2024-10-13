# 🤖 RightsBot: Your Constitutional Guide

RightsBot is a chatbot application built using **LangChain**, **FAISS**, **OpenAI GPT**, and **Streamlit**. It provides users with answers to legal questions based on the Constitution of Pakistan. The bot retrieves relevant sections from a pre-built vectorstore of the Constitution and generates fact-based responses using GPT-4.

## Features

- **Legal Expertise**: RightsBot acts as a legal expert on the Constitution of Pakistan, answering questions strictly based on its contents.
- **Conversational Memory**: The bot retains a limited conversation history, considering only the last two exchanges to maintain context.
- **Vector Search**: It uses FAISS for fast and accurate retrieval of constitutional clauses to support its answers.
- **Streamed Responses**: The bot provides streamed answers for faster interaction.
- **User-friendly Interface**: Built with Streamlit, the interface allows users to easily interact with the chatbot, clear chat history, and provide an OpenAI API key.

## Setup Instructions

### Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key**
3. **FAISS Vectorstore**: A pre-built FAISS index of the Constitution of Pakistan needs to be present.

### Installation

1. **Clone the repository**:

   bash
   git clone https://github.com/AdeelAfzal01/PakGPT.git

2. **Create a virtual environment and activate it**:

    python -m venv env
    source env/bin/activate   # On Windows use `env\Scripts\activate`

3. **pip install -r requirements.txt**

    pip install -r requirements.txt

4. **Set up FAISS vectorstore**

    Ensure you have the faiss_index_datamodel_bigchunks file ready. This file should be generated using FAISS and the Constitution of Pakistan's contents embedded using OpenAI embeddings.

### Usage

1. **Run the application**

    streamlit run PakGPT.py

2. **Provide OpenAI API Key**

    Upon launching, go to the sidebar.
    Enter your OpenAI API key to enable the chatbot.

3. **Interact with the bot**

    Ask questions such as:
    "What are the fundamental rights guaranteed to every Pakistani citizen?"
    "How can I protect my rights if they are violated?"

4. **Clear Chat History**

    Press the Clear Chat button to start a fresh conversation.

### Code Explanation
    The core functionality of the bot is defined as follows:

    LLM Setup: It uses GPT-4 (gpt-4o model) with a custom prompt that instructs the bot to respond like a legal expert.
    Memory Management: Conversation history is limited to the last two messages to reduce latency and avoid long-term context retention.
    Conversational Chain: The bot uses ConversationalRetrievalChain to retrieve relevant sections of the Constitution and use them in response generation.
    Streamlit Interface**: The front-end interface allows users to enter queries, view responses, and manage their API key.

### Future Improvements
    Multilingual Support: Add the ability for the bot to handle multiple languages, including Urdu.
    Advanced Retrieval: Improve retrieval by optimizing chunk sizes and adding more sophisticated document embedding techniques.










