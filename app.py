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

# TODO:
# - experiment with entity memory
# - experiment with multiple memory injection into chain (entity, conversation, summary)

def chat(user_input):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                f"This is a conversation between a human and an AI philosopher of Stoicism. "
                "You help the human reflect on the core concepts of Stoicism as though you are Epictetus himself. "
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    llm = ChatOpenAI(temperature=TEMPERATURE, model=MODEL)
    memory = ConversationSummaryMemory(llm=OpenAI(), return_messages=True)
    conversation = ConversationChain(
        memory=memory, prompt=prompt, llm=llm, verbose=True
    )

    return conversation.predict(input=user_input)


def main():
    st.title('Stoic Reflection with AI')
    st.write("Welcome to the Stoic reflection tool based on the principles in Epictetus's Enchiridion.")

    if not hasattr(st.session_state, 'messages'):
        st.session_state.messages = []
    user_input = st.text_input('Enter your reflection or question:')
    if user_input:
        st.session_state.messages.append(user_input)
        response = chat(user_input)
        st.session_state.messages.append(response)
    st.write(st.session_state.messages)


if __name__ == "__main__":
    main()
