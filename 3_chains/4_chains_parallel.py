from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda,RunnableParallel

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

summary_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a movie critic"),
        ("human","Provide a bride summary of the movie {movie_name}."),
    ]
)

#define plot analyze step

def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system","You are a movie critic"),
            ("human","Analayze the plot:{plot}. What are its strenghths and weakness?"),
        ]
    )
    return plot_template.format_prompt(plot=plot)

def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system","You are a movie critic"),
            ("human","Analayze the characters:{characters}. What are their strenghths and weakness?"),
        ]
    )
    return character_template.format_messages(characters=characters)


def combine_verdicts(plot_analaysis,character_analaysis):
    return f"Plot Analaysis:\n{plot_analaysis}\n\nCharacter Analysis:\n{character_analaysis}"

plot_brach_chain = (
    RunnableLambda(lambda x:analyze_plot(x))|model|StrOutputParser()
)

character_brach_chain = (
    RunnableLambda(lambda x:analyze_characters(x))|model|StrOutputParser()
)

chain = (
    summary_template
    |model
    |StrOutputParser()
    |RunnableParallel(branches={"plot":plot_brach_chain,"characters":character_brach_chain})
    |RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"],x["branches"]["characters"]))   
)

result = chain.invoke({"movie_name":"Pushpa"})

print(result)