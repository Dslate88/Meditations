import json
import random
import argparse
from colorama import Fore, Style

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

TEMPERATURE = 0.5
MODEL = "gpt-4"

def chat():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            f"This is a conversation between a human and an AI philosopher of Stoicsm. "
            "You help the human reflect on the core concepts of Stoicism as though you are Epictetus himself. "
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    llm = ChatOpenAI(temperature=TEMPERATURE, model=MODEL)
    conversation_with_summary = ConversationChain(
        llm=llm,
        memory=ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40, return_messages=True),
        verbose=True
    )

    # Starts the chat conversation
    print('Start chatting (type "quit" to exit):\n')
    # print concept, definition, chapter, question
    # print(f"{Fore.GREEN}Concept: {Style.RESET_ALL}{concept['title']}")
    # print(f"{Fore.GREEN}Definition: {Style.RESET_ALL}{concept['definition']}")
    # print(f"{Fore.GREEN}Chapter: {Style.RESET_ALL}{chapter}")
    # print(f"{Fore.GREEN}Question: {Style.RESET_ALL}{question}\n")

    while True:
        user_input = input(f'{Fore.BLUE}> {Style.RESET_ALL}')
        if user_input.lower() == 'quit':
            break

        resp = conversation_with_summary.predict(input=user_input)
        print(f'{Fore.MAGENTA}{resp}{Style.RESET_ALL}\n')

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="A tool for Stoic reflection based on the principles in Epictetus's Enchiridion.")
    parser.add_argument('--action', required=True, choices=['reflection'], help="Action to perform. Currently supports 'reflection' only.")
    # parser.add_argument('--type', required=True, choices=['random'], help="Type of action to perform. Currently supports 'random' only.")

    args = parser.parse_args()

    if args.action == 'reflection':
        chat()

if __name__ == "__main__":
    main()


