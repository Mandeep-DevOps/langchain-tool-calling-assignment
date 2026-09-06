# ============================================================
# LangChain Tool Calling Assignment
# Base Version - Before adding the best_food tool
# ============================================================

# Import the @tool decorator from LangChain.
# A decorator modifies or extends the behavior of a Python function.
# Here, @tool converts a normal Python function into a tool
# that the LLM can understand and request to use.
from langchain.tools import tool


# Import load_dotenv().
# This allows Python to read environment variables stored
# inside a .env file.
#
# Later, our OpenAI API key will be stored there.
from dotenv import load_dotenv


# Import ChatOpenAI.
# This is the LangChain class that lets our Python program
# communicate with an OpenAI chat model.
from langchain_openai import ChatOpenAI


# Import LangChain message classes.
#
# HumanMessage = message coming from the user.
# AIMessage    = message produced by the AI model.
# SystemMessage = instructions that can be given to the model.
# ToolMessage   = result returned by a tool back to the model.
#
# AIMessage and SystemMessage are imported because they were
# present in the trainer's original code.
# They are not directly used in this version yet.
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)


# Import Python's built-in JSON module.
#
# JSON is a common format used for exchanging structured data
# between applications.
import json


# ============================================================
# Load environment variables
# ============================================================

# Look for a .env file and load the variables stored inside it.
#
# We will later create:
#
# OPENAI_API_KEY=...
#
# This prevents us from writing the API key directly in Python code.
load_dotenv()


# ============================================================
# TOOL 1 - Weather Tool
# ============================================================

# @tool tells LangChain:
# "This Python function is available for the LLM to use."
@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.

    Args:
        city: The name of the city.
    """

    # weather_data is a Python dictionary.
    #
    # A dictionary stores information as:
    #
    # key : value
    #
    # Example:
    #
    # "bangalore" : "Sunny, 28°C"
    #
    # "bangalore" is the key.
    # "Sunny, 28°C" is the value.
    #
    # This is MOCK data.
    # We are not actually calling a weather website or API.
    weather_data = {
        "bangalore": "Sunny, 28°C",
        "mumbai": "Rainy, 26°C",
        "delhi": "Cloudy, 22°C"
    }

    # city.lower()
    #
    # Converts the city entered by the user to lowercase.
    #
    # Example:
    #
    # Bangalore
    #     |
    #     v
    # bangalore
    #
    # This helps match the lowercase keys in our dictionary.
    #
    # dictionary.get(key, default_value)
    #
    # searches for the city.
    # If the city does not exist, it returns the fallback message.
    return weather_data.get(
        city.lower(),
        "Weather data not available"
    )


# ============================================================
# TOOL 2 - Flight Booking Tool
# ============================================================

@tool
def book_flight(origin: str, destination: str, date: str) -> dict:
    """
    Book a flight from one city to another.

    Args:
        origin: The origin city.
        destination: The destination city.
        date: The date of the flight.
    """

    # This is also a MOCK implementation.
    #
    # A real production application would normally call
    # an airline API or booking service.
    #
    # Here we simply return a Python dictionary.
    return {
        "booking_id": "1234567890",

        # f-string:
        #
        # f"{origin} to {destination}"
        #
        # lets us insert Python variables inside a string.
        #
        # Example:
        #
        # origin = Bangalore
        # destination = Mumbai
        #
        # Result:
        #
        # Bangalore to Mumbai
        "route": f"{origin} to {destination}",

        "date": date,
        "status": "confirmed"
    }


# ============================================================
# Register the available tools
# ============================================================

# A Python list stores multiple items in order.
#
# Currently our application has TWO tools:
#
# 1. get_weather
# 2. book_flight
#
# Later in the assignment we will add:
#
# 3. best_food
tools = [
    get_weather,
    book_flight
]


# ============================================================
# Create a lookup table for our tools
# ============================================================

# This is called a dictionary comprehension.
#
# It creates a dictionary where:
#
# key   = tool name
# value = actual tool object
#
# Conceptually it becomes something like:
#
# {
#     "get_weather": get_weather,
#     "book_flight": book_flight
# }
#
# We need this later because the LLM will tell us the NAME
# of the tool it wants to execute.
tool_names = {
    t.name: t
    for t in tools
}


# Display the registered tools.
print("Tool Names:", tool_names)

print("*" * 70)


# ============================================================
# Create the LLM
# ============================================================

# ChatOpenAI creates the language model object.
#
# model:
# Selects the OpenAI model.
#
# temperature:
# Controls randomness in model responses.
#
# bind_tools(tools):
# Tells the model which tools are available.
#
# IMPORTANT:
#
# bind_tools() does NOT execute the tools.
#
# It only tells the LLM:
#
# "These are the tools you are allowed to request."
llm_openai = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=1
).bind_tools(tools)


print("*" * 70)


# ============================================================
# Main execution function
# ============================================================

# def creates a Python function.
#
# run is the function name.
#
# prompt: str
#
# means that prompt is expected to be a string.
def run(prompt: str):

    print("*" * 70)

    print("Prompt:", prompt)


    # ========================================================
    # STEP 1 - Send the user's question to the LLM
    # ========================================================

    # HumanMessage represents the user's request.
    #
    # invoke() sends the message to the LLM.
    msg = llm_openai.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )


    # msg.content contains normal text returned by the model.
    print("Model Output:", msg.content)


    # msg.tool_calls contains any tool the model wants us to execute.
    #
    # For example, the model might generate:
    #
    # tool = book_flight
    #
    # arguments:
    #
    # origin = Bangalore
    # destination = Mumbai
    # date = 12/01/2026
    print("Model Tool Calls:", msg.tool_calls)


    print("*" * 70)

    print("Model Full Output:", msg)

    print("*" * 70)


    # ========================================================
    # STEP 2 - Build conversation history
    # ========================================================

    # A Python list stores the conversation messages.
    #
    # We include:
    #
    # 1. Original user question
    # 2. Model response requesting a tool
    #
    # This is important because the model must later know
    # which tool it requested.
    messages = [
        HumanMessage(content=prompt),
        msg
    ]


    # ========================================================
    # STEP 3 - Execute the requested tool
    # ========================================================

    # A for loop processes every tool call generated by the model.
    #
    # Usually there may be one tool call,
    # but the model can potentially request multiple tools.
    for call in msg.tool_calls:

        # Extract the tool name.
        #
        # Example:
        #
        # "book_flight"
        tool_name = call["name"]


        # Extract arguments generated by the LLM.
        #
        # Example:
        #
        # {
        #     "origin": "Bangalore",
        #     "destination": "Mumbai",
        #     "date": "12/01/2026"
        # }
        tool_args = call["args"]


        # Every tool call receives a unique ID.
        #
        # LangChain uses this ID to connect the returned
        # ToolMessage with the correct original tool request.
        tool_id = call["id"]


        # ====================================================
        # Execute the Python tool
        # ====================================================

        # tool_names[tool_name]
        #
        # finds the actual Python tool using its name.
        #
        # .invoke(tool_args)
        #
        # executes that tool using the arguments selected
        # by the LLM.
        tool_result = tool_names[tool_name].invoke(tool_args)


        print("Tool Result:", tool_result)

        print("*" * 70)


        # ====================================================
        # Send the tool result back into conversation history
        # ====================================================

        # ToolMessage represents the output of an executed tool.
        #
        # json.dumps() converts the result into JSON-compatible text.
        messages.append(
            ToolMessage(
                content=json.dumps(tool_result),
                tool_call_id=tool_id
            )
        )


    print("Messages:", messages)

    print("*" * 70)


    # ========================================================
    # STEP 4 - Ask the LLM to create the final answer
    # ========================================================

    # We send the complete conversation back to the model:
    #
    # User Question
    #       +
    # Model Tool Request
    #       +
    # Tool Result
    #
    # The model can now generate a natural-language response.
    final_msg = llm_openai.invoke(messages)


    print("Final Model Output:", final_msg.content)


# ============================================================
# Run the program
# ============================================================

# The weather example from the trainer can be enabled later.
#
# run("What is the weather in Bangalore?")


# For now, use the original flight-booking test.
run("Book a flight from Bangalore to Mumbai on 12/01/2026")