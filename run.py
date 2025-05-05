import streamlit as st
import pandas as pd
import re
from langchain_community.chat_models import ChatOllama
from langchain.agents import AgentType
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os, glob, hashlib

# Directory where charts are saved
chart_dir = "D:/llm-vehicle-tracking-dashboard/charts"
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

    # Prefix instruction for agent
    prefix = """
        You are a data science expert and visualization specialist analyzing vehicle tracking CSV data using pandas, numpy, matplotlib.pyplot, and seaborn. The DataFrame 'df' is already loaded. Do not use pd.read_csv().
    
        Strict Guidelines:
        - Only return Python code that generates visual charts.
        - Always import pandas, numpy, matplotlib.pyplot, and seaborn at the top.
        - Always use "vehicle_reg_no" column for vehicle numbers, df doesn't have the "vehicle_id"
        - Always convert 'vehicle_timestamp' using:
            df['vehicle_timestamp'] = pd.to_datetime(df['vehicle_timestamp'], errors='coerce')
        - Always use df.sort_values(by='vehicle_timestamp') before any time-based operation.
        - Never convert 'vehicle_timestamp' into the index unless absolutely required.
        - Never use .dt accessor on an index — only use it on datetime columns.
        - To calculate time differences in hours, always use:
            .diff().dt.total_seconds() / 3600
        - Always filter by vehicle number using df[df['vehicle_reg_no'] == '<VEHICLE_NO>'] BEFORE any groupby or resample operation.
        - Never group by ['date', 'vehicle_reg_no']; instead filter first, then group by date only.
        - Use only numeric aggregations in groupby.
        - Do not apply sum/mean on non-numeric columns.
        - Plot only using matplotlib or seaborn with professional styling:
            - Clear titles, axis labels, grid
            - Label x-axis with actual timestamp values, not synthetic ranges
            - Use readable figure sizes and tight layout
        
        Plotting Rules:
        - Always use "vehicle_reg_no" column for vehicle numbers, df doesn't have the "vehicle_id"
        - For time trends (e.g. speed vs. time): plot line charts with datetime x-axis.
        - For daily travel durations: group by df['vehicle_timestamp'].dt.date and plot total hours.
        - For coordinates: use scatterplots or heatmaps if latitude and longitude are needed.
        - All plots must be visual, annotated, and self-explanatory.
        - Never write to or read from disk.
        - Do not create functions. Use a single, flat Python script.
        
        Error Resolution:
        - Always validate columns exist before use.
        - Ensure all groupby/resample operations use a proper datetime column, not an invalid index.
        - Handle NaNs or missing timestamps gracefully using `.fillna(0)` when needed.
        - If a column like 'vehicle_reg_no' or 'vehicle_timestamp' is missing, return: "I don't know".
        
        Return ONLY a complete Python code block that generates a visual chart based on user query.

    """

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
