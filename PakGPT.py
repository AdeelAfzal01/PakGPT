import streamlit as st
import openai
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import warnings
warnings.filterwarnings("ignore")



# Parameter
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_VECTORSTORE = "faiss_index_datamodel_bigchunks"
OPENAI_LLM_MODEL = "gpt-4o"

# Sidebar content: prompt user to input their OpenAI API key
with st.sidebar:
    st.title('🤖 PakGPT Bot : Your Constitutional Guide')
    st.markdown("Please enter your OpenAI API key to start using the chatbot.")

    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    if openai_api_key:
        st.session_state.api_key = openai_api_key

# Initial assistant message
initial_message = """
Hi there! I'm your PakGPT Bot 🤖 
Here are some questions you might ask me:\n
❓ What are the fundamental rights guaranteed to every Pakistani citizen?\n
❓ How can I register to vote in Pakistan?\n
❓ What are my rights as a citizen in Pakistan?\n
❓ What are the economic rights of Pakistani citizens?\n
❓ How can I protect my rights if they are violated?
"""

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
st.button('Clear Chat', on_click=clear_chat_history)

# Only proceed if the user has provided an OpenAI API key
if "api_key" in st.session_state:
    # Load the vectorstore and set up the LLM
    try:

        openai.api_key = st.session_state.api_key

        # Load the embedding model and vectorstore
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        persisted_vectorstore = FAISS.load_local(LOCAL_VECTORSTORE, embeddings, allow_dangerous_deserialization=True)

        # Chat history
        class LimitedConversationBufferMemory(ConversationBufferMemory):
            def save_context(self, inputs, outputs):
                super().save_context(inputs, outputs)
                # Limit the chat history to the last 3 messages
                self.chat_memory.messages = self.chat_memory.messages[-2:]
        # Use the custom memory class
        memory = LimitedConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        
        # Retriever
        retriever = persisted_vectorstore.as_retriever(search_kwargs={"k":3})

        # LLM
        streaming_callback = StreamingStdOutCallbackHandler()
        llm = ChatOpenAI(model_name=OPENAI_LLM_MODEL, temperature=0, max_tokens=2500, timeout=30, streaming=True, callbacks=[streaming_callback])

        # Set up the conversation chain
        prompt_template = """
        You are a legal expert on the Constitution of Pakistan. Answer questions based only on its content. Follow these guidelines:

        1. Stay relevant: Answer only what is asked without assumptions.
        2. Stick to the Constitution: Politely refuse questions beyond its scope.
        3. Ensure accuracy: Verify answers with the Constitution.
        4. Be clear: Explain legal terms simply.
        5. No speculation: Provide facts only.
        6. Response should be less than 400 words and should be easy.
        7. If the user ask about the previous response from the LLM then don't use the context in that question but use only previous questions and answers.
        8. If the user says good bye message then respond with a happing ending message.
        9. Summarize the previous conversation.
    
        Previous Conversations: {chat_history}

        Constitution Context: {context}

        Query: {question}

        Answer:
        """
        custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )

        # User-provided prompt
        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate a new response if last message is not from assistant
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Hold on, I'm fetching the information for you..."):
                        response = conversation_chain({"question": prompt})
                        full_response = response['answer']
                        st.markdown(full_response)

                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.warning("Please enter your OpenAI API key in the sidebar to start using the chatbot.")
