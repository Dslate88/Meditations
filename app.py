import os
import uuid

import boto3
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
from langchain.memory import ConversationSummaryMemory, DynamoDBChatMessageHistory, ConversationBufferMemory

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

dynamodb = boto3.resource('dynamodb')

def check_existing_session(session_id):
    table = dynamodb.Table('SessionTable')
    response = table.get_item(Key={'SessionId': session_id})
    return 'Item' in response

def add_message_to_db(session_id, role, content):
    history = DynamoDBChatMessageHistory(table_name="SessionTable", session_id=session_id)
    if role == "user":
        history.add_user_message(content)
    else:
        history.add_ai_message(content)

def get_chat_history_from_db(session_id):
    table = dynamodb.Table('SessionTable')
    response = table.get_item(Key={'SessionId': session_id})
    messages = []
    if 'Item' in response and 'messages' in response['Item']:
        for msg in response['Item']['messages']:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            else:
                messages.append(AIMessage(content=msg['content']))
    return messages



class ChatSession:
    def __init__(self, model, temperature, system_message, session_id):
        self.MODEL = model
        self.TEMPERATURE = temperature
        self.SYSTEM_MESSAGE = system_message
        self.session_id = session_id
        self.conversation = self._setup_conversation()

        self.message_history = DynamoDBChatMessageHistory(
            table_name="SessionTable", session_id=session_id
        )

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

        memory = ConversationBufferMemory(
            memory_key="history", return_messages=True
        )

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

    if not hasattr(st.session_state, 'session_id'):
        st.session_state.session_id = str(uuid.uuid4())

    # Create chat instance on-the-fly:
    chat_instance = ChatSession(
        st.session_state.model,
        st.session_state.temperature,
        st.session_state.system_message,
        st.session_state.session_id
    )

    # Load previous messages from DynamoDB, if any
    if not hasattr(st.session_state, "messages"):
        st.session_state.messages = get_chat_history_from_db(st.session_state.session_id)


    # Display the conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        add_message_to_db(st.session_state.session_id, "user", prompt)
        response = chat_instance.get_response(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        add_message_to_db(st.session_state.session_id, "assistant", response)

if __name__ == "__main__":
    main()
