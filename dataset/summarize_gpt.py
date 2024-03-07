import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import LLMChain
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate

class summarize_gpt():
    def __init__(
        self,
        chatai_parms={},
        pre_prompt="",
        **kwargs,
    ):
        _ = load_dotenv(find_dotenv()) 
        self.llm = ChatOpenAI(**chatai_parms)

        with open(pre_prompt) as file:
            self.pre_prompt = file.read()

        self.create_inital_prompt()
        print()


    def create_inital_prompt(self,):
        self.inital_message = [
        SystemMessagePromptTemplate.from_template(self.pre_prompt),
        HumanMessage(content="""
                            Rock Hydra Creature — Hydra {X}{R}{R} Rock Hydra enters the battlefield with X +1/+1 counters on it.
                            For each 1 damage that would be dealt to Rock Hydra, if it has a +1/+1 counter on it,
                            remove a +1/+1 counter from it and prevent that 1 damage. {R}: Prevent the next 1 damage
                            that would be dealt to Rock Hydra this turn. {R}{R}{R}: Put a +1/+1 counter on Rock Hydra.
                            Activate only during your upkeep. P/T 0/0
                     """.replace('\n', ' ').strip(), 
                    example=True
                    ),
        AIMessage(
                    content="""
                            Rock Hydra Creature — Hydra {X}{R}{R} enters the battlefield with X +1/+1 counters. Removes a +1/+1 counter
                            to prevent 1 damage. {R}: Prevents 1 damage this turn. {R}{R}{R}: Add a +1/+1 counter during
                            upkeep. P/T 0/0
                    """.replace('\n', ' ').strip(),
                    example=True
                    ),
        ]

    def summarize_description(self, description):
        
        query = self.inital_message + [HumanMessage(content=description)]
        prompt_template = ChatPromptTemplate.from_messages(query)
        llm_chain = LLMChain(llm=self.llm, prompt=prompt_template)
        response = llm_chain.invoke({})
        shortened = response['text']

        return shortened