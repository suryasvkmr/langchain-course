import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Psych is an American detective comedy-drama television series created by Steve Franks for USA Network.[1] The series stars James Roday as Shawn Spencer, a young crime consultant for the Santa Barbara Police Department whose "heightened observational skills"[2] and impressive eidetic memory allow him to convince people that he solves cases with his psychic abilities. The program also stars Dulé Hill as Shawn's intelligent best friend and reluctant partner Burton "Gus" Guster, as well as Corbin Bernsen as Shawn's father Henry, a former detective with the Santa Barbara Police Department.[3]

    Psych premiered on July 7, 2006, following the fifth season premiere of Monk, and continued to be paired with the series until Monk's conclusion on December 4, 2009. During the second season, an animated segment titled "The Big Adventures of Little Shawn and Gus" was added to the series. Psych was the highest-rated American basic cable television premiere of 2006.[4] USA Network renewed the series for an eighth season on December 19, 2012, to include eight episodes, and ordered two more episodes on June 25, 2013, bringing the episode order to ten.[5][6] On February 5, 2014, USA Network confirmed that the eighth season of Psych would be its last, with the series finale airing on March 26, 2014.[7]

    sych: The Movie, a two-hour television film, aired on USA Network on December 7, 2017, launching the Psych film series,[8] with Franks's hope being to make five more Psych movies following Psych: The Movie.[9] On February 14, 2019, it was announced Psych: The Movie 2 was greenlit and set to premiere in late 2019, for which the main cast would return, but the premiere thereof was subsequently delayed to 2020, with the film renamed Psych 2: Lassie Come Home, and released on NBCUniversal's streaming service, Peacock, July 15, 2020, the day the service officially launched.[10][11][12] On May 13, 2021, Peacock announced a third film, Psych 3: This Is Gus, which premiered on November 18, 2021.[13][14] Three further Psych films were reportedly in development.[9]
    """
    summary_template = """
    given the following information {information}, create:
    1. a short summary
    2. two interesting facts about it
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )
    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.1-flash-lite")
    #llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
