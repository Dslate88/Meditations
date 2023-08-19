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
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory

TEMPERATURE = 0.5
MODEL = "gpt-3.5-turbo"


class StoicChat:
    def __init__(self):
        # Initialization of constants
        self.TEMPERATURE = 0.5
        self.MODEL = "gpt-3.5-turbo"

        # Load or initialize the conversation
        self.conversation = self._get_or_create_conversation()

    def _get_or_create_conversation(self):
        """Private method to create or retrieve an existing conversation."""
        if "conversation" not in st.session_state:
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        f"This is a conversation between a human and an AI philosopher of Stoicism. "
                        "You help the human reflect on the core concepts of Stoicism as though you are Epictetus himself."
                    ),
                    MessagesPlaceholder(variable_name="history"),
                    HumanMessagePromptTemplate.from_template("{input}"),
                ]
            )

            llm = ChatOpenAI(temperature=self.TEMPERATURE, model=self.MODEL)
            memory = ConversationSummaryMemory(llm=OpenAI(), return_messages=True)

            st.session_state.conversation = ConversationChain(
                memory=memory, prompt=prompt, llm=llm, verbose=True
            )

        return st.session_state.conversation

    def get_response(self, user_input):
        """Public method to get a response based on user input."""
        return self.conversation.predict(input=user_input)


def main():
    st.title("Stoic Reflection with AI")
    st.write(
        "Welcome to the Stoic reflection tool based on the principles in Epictetus's Enchiridion."
    )

    chat_instance = StoicChat()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = chat_instance.get_response(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
