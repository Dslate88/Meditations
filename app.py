import os

import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory, PostgresChatMessageHistory

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# DATABASE_URL = os.environ.get("DATABASE_URL")
DATABASE_URL = "postgresql://postgres:mypassword@localhost/chat_history"


class ChatSession:
    def __init__(self, model, temperature, system_message):
        self.MODEL = model
        self.TEMPERATURE = temperature
        self.SYSTEM_MESSAGE = system_message
        self.conversation = self._setup_conversation()

    def _setup_conversation(self):
        """Initialize the conversation based on provided settings."""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.SYSTEM_MESSAGE),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

        llm = ChatOpenAI(temperature=self.TEMPERATURE, model=self.MODEL)
        memory = ConversationSummaryMemory(llm=OpenAI(), return_messages=True)

        return ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=True)

    def get_response(self, user_input):
        """Public method to get a response based on user input."""
        return self.conversation.predict(input=user_input)


def main():
    st.title("Stoic Reflection with AI")
    st.write(
        "Welcome to the Stoic reflection tool based on the principles in Epictetus's Enchiridion."
    )

    # Sidebar Configuration
    st.sidebar.header("Configuration")

    model = st.sidebar.selectbox(
        "Select AI Model:", ["gpt-3.5-turbo", "gpt-4"], index=0
    )

    temperature = st.sidebar.slider(
        "Temperature (Determines Randomness)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    default_system_msg = (
        "This is a conversation between a human and an AI philosopher of Stoicism. "
        "You help the human reflect on the core concepts of Stoicism as though you are Epictetus himself."
    )

    system_message = st.sidebar.text_area("System Message:", value=default_system_msg)

    if not hasattr(st.session_state, "temperature"):
        print("Initializing temperature")
        st.session_state.temperature = temperature

    if not hasattr(st.session_state, "model"):
        print("Initializing model")
        st.session_state.model = model

    if not hasattr(st.session_state, "system_message"):
        print("Initializing system message")
        st.session_state.system_message = system_message

    # If any sidebar setting has changed, update the state
    st.session_state.temperature = temperature
    st.session_state.model = model
    st.session_state.system_message = system_message

    # Initialize chat instance using the state values
    if not hasattr(st.session_state, "chat_instance"):
        print("Initializing chat instance")
        st.session_state.chat_instance = ChatSession(
            st.session_state.model,
            st.session_state.temperature,
            st.session_state.system_message,
        )

    # Display the conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = st.session_state.chat_instance.get_response(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
