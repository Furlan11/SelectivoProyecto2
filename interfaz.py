import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import os
import datetime

# Cargar la clave API desde el archivo .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# Función para guardar el historial de consultas
def save_history(question, answer, agent_type, agent):
    with open("history.txt", "a") as f:
        f.write(f"{datetime.datetime.now()}: [{agent_type} - {agent}] {question} -> {answer}\n")


# Función para cargar el historial
def load_history():
    if os.path.exists("history.txt"):
        with open("history.txt", "r") as f:
            return f.readlines()
    return []


# Crear agentes
def create_agents():
    base_prompt = hub.pull("langchain-ai/react-agent-template")

    # Agente principal de Python
    python_instructions = """You are an agent designed to write and execute Python code to answer questions."""
    python_prompt = base_prompt.partial(instructions=python_instructions)
    python_tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=python_prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo", openai_api_key=api_key),
        tools=python_tools,
    )
    python_executor = AgentExecutor(agent=python_agent, tools=python_tools, verbose=True)

    # Agente de Transposición de Matrices
    matrix_transpose_instructions = """You are an agent designed to calculate the transpose of matrices."""
    matrix_transpose_prompt = base_prompt.partial(instructions=matrix_transpose_instructions)
    matrix_transpose_agent = create_react_agent(
        prompt=matrix_transpose_prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo", openai_api_key=api_key),
        tools=[PythonREPLTool()],
    )
    matrix_transpose_executor = AgentExecutor(agent=matrix_transpose_agent, tools=[PythonREPLTool()], verbose=True)

    # Agente para Derivadas
    derivative_instructions = """You are an agent designed to calculate derivatives of mathematical expressions."""
    derivative_prompt = base_prompt.partial(instructions=derivative_instructions)
    derivative_agent = create_react_agent(
        prompt=derivative_prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo", openai_api_key=api_key),
        tools=[PythonREPLTool()],
    )
    derivative_executor = AgentExecutor(agent=derivative_agent, tools=[PythonREPLTool()], verbose=True)

    # Agente para Resolver Ecuaciones Algebraicas
    equation_solver_instructions = """You are an agent designed to solve algebraic equations symbolically."""
    equation_solver_prompt = base_prompt.partial(instructions=equation_solver_instructions)
    equation_solver_agent = create_react_agent(
        prompt=equation_solver_prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo", openai_api_key=api_key),
        tools=[PythonREPLTool()],
    )
    equation_solver_executor = AgentExecutor(agent=equation_solver_agent, tools=[PythonREPLTool()], verbose=True)

    # Crear agentes CSV
    csv_files = {
        "Quotes": "quotes.csv",
        "Planets": "planets.csv",
        "Events": "events.csv",
        "Films": "films.csv",
        "Characters": "characters.csv",
        "Vehicles": "vehicles.csv",
    }
    csv_agents = {}
    for name, path in csv_files.items():
        csv_agents[name] = create_csv_agent(
            llm=ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=api_key),
            path=path,
            verbose=True,
            allow_dangerous_code=True,
        )

    return {
        "Python Agents": {
            "Python Agent": python_executor,
            "Matrix Transpose Agent": matrix_transpose_executor,
            "Derivative Agent": derivative_executor,
            "Equation Solver Agent": equation_solver_executor,
        },
        "CSV Agents": csv_agents,
    }


# Selección automática para Python Agents
def decide_python_agent(user_input, python_agents):
    user_input_lower = user_input.lower()
    if "derivative" in user_input_lower or "differentiate" in user_input_lower:
        return "Derivative Agent"
    elif "matrix" in user_input_lower or "transpose" in user_input_lower:
        return "Matrix Transpose Agent"
    elif "solve" in user_input_lower or "equation" in user_input_lower:
        return "Equation Solver Agent"
    else:
        return "Python Agent"


# Selección automática para CSV Agents
def decide_csv_agent(user_input, csv_agents):
    user_input_lower = user_input.lower()
    for name in csv_agents.keys():
        if name.lower() in user_input_lower:
            return name
    return list(csv_agents.keys())[0]  # Predeterminado al primero si no coincide


# Aplicación principal
def main():
    st.set_page_config(page_title="Agente Interactivo", page_icon="🤖", layout="wide")
    st.title("🤖 Agente Interactivo")
    st.markdown("""
        Este agente puede realizar las siguientes tareas:
        - Ejecutar y depurar código Python (incluyendo cálculos específicos como derivadas y resolver ecuaciones).
        - Consultar datos en archivos CSV.
    """)

    # Crear agentes
    agents = create_agents()

    # Apartado para Python Agents
    st.header("Python Agents")
    python_input = st.text_area("Consulta para Python Agents:", placeholder="Ej. Transpone la matriz [[1, 2], [3, 4]].")
    auto_detect_python = st.checkbox("Seleccionar Python Agent automáticamente", value=True)

    if not auto_detect_python:
        selected_python_agent = st.selectbox("Selecciona un Python Agent específico:",
                                             list(agents["Python Agents"].keys()))
    else:
        selected_python_agent = None

    python_execute = st.button("Ejecutar con Python Agent")

    if python_execute and python_input:
        try:
            if auto_detect_python:
                selected_python_agent = decide_python_agent(python_input, agents["Python Agents"])
            st.markdown(f"### Usando el Python Agent: {selected_python_agent}")
            python_agent = agents["Python Agents"][selected_python_agent]
            response = python_agent.invoke({"input": python_input})
            st.markdown("### Respuesta del Python Agent:")
            st.code(response["output"], language="python")
            save_history(python_input, response["output"], "Python", selected_python_agent)
        except Exception as e:
            st.error(f"Error con {selected_python_agent}: {e}")

    # Apartado para CSV Agents
    st.header("CSV Agents")
    csv_agents = agents["CSV Agents"]
    csv_input = st.text_area("Consulta para CSV Agents:", placeholder="Ej. ¿Qué personaje dice 'I am your father'?")
    auto_detect_csv = st.checkbox("Seleccionar CSV Agent automáticamente", value=True)

    if not auto_detect_csv:
        selected_csv_agent = st.selectbox("Selecciona un CSV Agent específico:", list(csv_agents.keys()))
    else:
        selected_csv_agent = None

    csv_execute = st.button("Ejecutar con CSV Agent")

    if csv_execute and csv_input:
        try:
            if auto_detect_csv:
                selected_csv_agent = decide_csv_agent(csv_input, csv_agents)
            st.markdown(f"### Usando el CSV Agent: {selected_csv_agent}")
            csv_agent = csv_agents[selected_csv_agent]
            response = csv_agent.invoke({"input": csv_input})
            st.markdown("### Respuesta del CSV Agent:")
            st.code(response["output"])
            save_history(csv_input, response["output"], "CSV", selected_csv_agent)
        except Exception as e:
            st.error(f"Error con {selected_csv_agent}: {e}")

    # Mostrar historial
    if st.checkbox("Mostrar historial"):
        history = load_history()
        if history:
            st.markdown("### Historial de consultas:")
            st.text("".join(history))
        else:
            st.info("No hay historial aún.")


# Iniciar la aplicación
if __name__ == "__main__":
    main()
