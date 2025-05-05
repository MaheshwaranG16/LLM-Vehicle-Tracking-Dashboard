import streamlit as st
import pandas as pd
import re
from langchain_community.chat_models import ChatOllama
from langchain.agents import AgentType
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os, glob, hashlib

# Directory where charts are saved
chart_dir = "/home/ubuntu/LLM-Vehicle-Tracking-Dashboard/charts"
cwd = os.getcwd()

# Clean up old .png files before generating new ones
old_files = glob.glob(os.path.join(chart_dir, "*.png"))
for file in old_files:
    os.remove(file)

# Initialize the LLM model from Ollama
llm = ChatOllama(model="mistral")

# Streamlit UI
st.title("Vehicle Tracking System Analysis")
st.write("Upload your Vehicle Tracking Data CSV")

file = st.file_uploader("Select your file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    # Convert vehicle_timestamp column to datetime right after loading
    if 'vehicle_timestamp' in df.columns:
        df['vehicle_timestamp'] = pd.to_datetime(df['vehicle_timestamp'], errors='coerce')

    # Pass df to PythonREPLTool so that it can be used directly
    tools = [PythonREPLTool(locals={"df": df})]

    # Path to your prefix text file in the same directory
    prefix_file_path = os.path.join(chart_dir, "prefix.txt")

    # Load the prefix from the text file
    with open(prefix_file_path, "r") as file:
        prefix = file.read()

    input_query = st.text_area("Ask your question related to Vehicle Tracking Data")

    if input_query:
        button = st.button("Submit")
        if button:
            # Create the agent with the given instructions and tools
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                prefix=prefix,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True,
                extra_tools=tools
            )

            # Invoke the agent with user input
            response = agent.invoke(input_query)

            # Extract and display only the Python code
            code_blocks = re.findall(r"```python(.*?)```", response["output"], re.DOTALL)
            if code_blocks:
                code = code_blocks[0].strip()
                with st.expander("Click to view generated Python code"):
                    st.code(code, language="python")

                # Unique filename based on query
                filename_hash = hashlib.md5(input_query.encode()).hexdigest()[:8]
                filename = f"chart_{filename_hash}.png"
                save_path = os.path.join(chart_dir, filename)

                # Replace plt.show() with savefig
                code = code.replace("plt.show()", f'plt.savefig(r"{save_path}")')

                # Execute the code safely in a local context
                local_context = {"df": df}
                try:
                    exec(code, {}, local_context)
                    result = local_context.get("result", None)
                    if result is not None:
                        st.write("Result:", result)
                except Exception as e:
                    st.error(f"Error while executing code: {e}")
            else:
                st.write(response["output"])  # fallback if no code found

            # Display all generated chart images
            new_files = glob.glob(os.path.join(chart_dir, "*.png"))
            if new_files:
                for file_path in new_files:
                    st.image(file_path)
