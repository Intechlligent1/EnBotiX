import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
    Answer the question below.
    Here is the conversation history: {context}
    
    Question: {question}

    Answer:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation(context, user_input):
    result = chain.invoke({"context": context, "question": user_input})
    return result

def main():
    st.title("IntechXAI Chatbot 🤖")
    st.write("Welcome to the IntechXAI Chatbot. Type your questions below!")

    if "context" not in st.session_state:
        st.session_state.context = ""

    user_input = st.text_input("You:", key="user_input")

    if user_input:
        if user_input.lower() == "exit":
            st.write("Goodbye!")
            st.stop()
  
        bot_response = handle_conversation(st.session_state.context, user_input)
 
        st.write(f"Bot: {bot_response}")
        # with st.sidebar:
        #     st.write(" Hello")

        st.session_state.context += f"\nUser: {user_input}\nAI: {bot_response}"

    st.write("### Conversation History")
    st.text(st.session_state.context)

if __name__ == "__main__":
    main()