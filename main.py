import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

load_dotenv()



def main():
    print("Hello from langchain-course!")
    information = """Muthuvel Karunanidhi Stalin[a] (born 1 March 1953) is an Indian politician and statesman who served as the eighth chief minister of Tamil Nadu from 2021 to 2026.[1] He became president of the Dravida Munnetra Kazhagam (DMK) on 28 August 2018, after serving as the party's working president from January 2017 to August 2018.
                    Stalin is the third son of former Tamil Nadu chief minister M. Karunanidhi. He completed his education at Presidency College, Madras in 1973. He began his political career in the late 1960s and was elected to the DMK's general committee in 1973. He was jailed during the Emergency in 1976. He became the secretary of the party's youth wing in 1982, a post he held for more than four decades. He served as the mayor of Chennai from 1996 to 2002. He has been elected to the Tamil Nadu Legislative Assembly eight times, and served as the state's first deputy chief minister from 2009 to 2011.
                    Stalin is married to Durga, and their son Udhayanidhi served as the deputy chief minister of Tamil Nadu from 2024 to 2026. In 2009, Anna University conferred an honorary doctorate on Stalin. In 2025, The Indian Express named him as India's 23rd most powerful personality.
                    Muthuvel Karunanidhi Stalin was born on 1 March 1953 in Madras. He is the third son of M. Karunanidhi, who would later serve as Chief Minister of Tamil Nadu, and Dayalu Ammal. He was named after the Soviet leader Joseph Stalin.[2][3] Stalin was educated at Madras Christian College Higher Secondary School.[4] He completed a pre-university course at Vivekananda College and obtained a degree in history from Presidency College in 1973.[5]
                    Stalin married Durga (alias Shantha) on 20 August 1975. They have two children:[5] a son Udhayanidhi, who served as deputy chief minister of Tamil Nadu from 2024 till 2026.[6] and daughter Senthamarai, an entrepreneur and education professional.[7][8] Stalin describes himself as an atheist."""
    
    summary_templete = """
    given the information {information}, about a person I want you to create
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_templete = PromptTemplate(
        input_variables=["information"],
        template=summary_templete,
    )

    llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROK_API_KEY")

)

    chain = summary_prompt_templete | llm
    response = chain.invoke({"information": information})

    print("Summary and interesting facts:")
    print(response.content)

if __name__ == "__main__":
    main()
