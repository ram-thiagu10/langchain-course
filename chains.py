import datetime
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schemas import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI(
    model="openrouter/free",  # Corrected typo: openrouter/free
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about llm/ai based 6+ years experienced candiadte interview preparation guide"
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)

# if __name__ == "__main__":
#     # 1. The original question
#     user_query = "Write about MK Stalin's success and failuers"

#     # This list acts as our Reflexion "Memory Notebook"
#     reflexion_memory = []

#     # We will let the agent try 2 times to perfect its answer
#     max_iterations = 2
#     current_messages = [HumanMessage(content=user_query)]

#     for loop in range(max_iterations):
#         print(f"\n--- 🔄 ITERATION {loop + 1} ---")

#         if loop == 0:
#             # First attempt uses the initial responder
#             draft_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") | parser_pydantic
#             result = draft_chain.invoke(input={"messages": current_messages})
#         else:
#             # Revisions use the revisor and inject our "memory"
#             # We append the past critique into the prompt context
#             revision_chain = revisor | parser_pydantic
#             result = revision_chain.invoke(input={"messages": current_messages})

#         # Access the parsed Pydantic object fields safely
#         # result[0] because PydanticToolsParser returns a list of objects
#         agent_data = result[0]

#         print(f"🤖 CURRENT ANSWER:\n{agent_data.answer}\n")
#         print(f"🤔 SELF CRITIQUE (Missing):\n{agent_data.reflection.missing}")
#         print(f"🤔 SELF CRITIQUE (Superfluous):\n{agent_data.reflection.superfluous}")

#         # Save the critique to our Reflexion Memory
#         critique_summary = f"Attempt {loop+1} Critique: Missing: {agent_data.reflection.missing}. Too much: {agent_data.reflection.superfluous}"
#         reflexion_memory.append(critique_summary)

#         # Update messages for the next loop execution so the LLM sees its past work
#         current_messages.append(HumanMessage(content=f"Your previous output was: {agent_data.answer}. Here is your notebook of past mistakes: {reflexion_memory}"))
