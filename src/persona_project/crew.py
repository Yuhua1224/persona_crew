import os
import yaml
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import MySQLSearchTool, MCPServerAdapter
from persona_tools import PersonaBuilderTool, PersonaAnalysisTool, QuestionnaireTool

# Load Big Five questionnaire metadata

def load_questionnaire(path="data/questionnaire.yml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

questionnaire = load_questionnaire()

# MySQL table search tool
mysql_tool = MySQLSearchTool(
    db_uri=os.getenv("PERSONA_MYSQL_URI"),
    table_name=os.getenv("PERSONA_SURVEY_TABLE", "survey_answers"),
)

persona_builder_tool = PersonaBuilderTool()
persona_analysis_tool = PersonaAnalysisTool()
questionnaire_tool = QuestionnaireTool()

@CrewBase
class PersonaCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    mcp_server_params = {
        "url": os.getenv("GMAIL_MCP_ENDPOINT"),
        "transport": "sse"
    }

    @agent
    def brain_agent(self) -> Agent:
        return Agent(config=self.agents_config["brain_agent"], tools=[], allow_delegation=True, verbose=True)

    @agent
    def translation_agent(self) -> Agent:
        return Agent(config=self.agents_config["translation_agent"], verbose=True)

    @agent
    def database_agent(self) -> Agent:
        return Agent(config=self.agents_config["database_agent"], tools=[mysql_tool], verbose=True)

    @agent
    def persona_construction_agent(self) -> Agent:
        return Agent(config=self.agents_config["persona_construction_agent"], tools=[persona_builder_tool], verbose=True)

    @agent
    def simulator_agent(self) -> Agent:
        return Agent(config=self.agents_config["persona_simulator_agent"], tools=[questionnaire_tool], verbose=True)

    @agent
    def validation_agent(self) -> Agent:
        return Agent(config=self.agents_config["validation_agent"], tools=[persona_analysis_tool, questionnaire_tool], verbose=True)

    @agent
    def email_agent(self, tools) -> Agent:
        return Agent(config=self.agents_config["email_agent"], tools=tools, verbose=True)

    ##################################################################################################

    @task
    def translate(self) -> Task:
        return Task(config=self.tasks_config["translate"], agent=self.translation_agent())

    @task
    def query(self) -> Task:
        return Task(config=self.tasks_config["query"], context=[self.translate()], agent=self.database_agent())

    @task
    def construct(self) -> Task:
        return Task(config=self.tasks_config["construct"], context=[self.query()], agent=self.persona_construction_agent())

    @task
    def simulate(self) -> Task:
        return Task(
            config=self.tasks_config["simulate"],
            context=[self.construct()],
            inputs={"questionnaire": questionnaire},
            agent=self.simulator_agent()
        )

    @task
    def validate(self) -> Task:
        return Task(
            config=self.tasks_config["validate"],
            context=[self.simulate(), self.query()],
            inputs={"questionnaire": questionnaire},
            agent=self.validation_agent()
        )

    @task
    def send_email(self, tools) -> Task:
        return Task(config=self.tasks_config["send_email"], context=[self.validate()], agent=self.email_agent(tools), markdown=True)

    @crew
    def crew(self) -> Crew:
        with MCPServerAdapter(self.mcp_server_params) as mcp_tools:
            gmail_tools = [t for t in mcp_tools if t.name == "gmail_send"]
            return Crew(
                agents=[
                    self.brain_agent(),
                    self.translation_agent(),
                    self.database_agent(),
                    self.persona_construction_agent(),
                    self.simulator_agent(),
                    self.validation_agent(),
                    self.email_agent(gmail_tools)
                ],
                tasks=[
                    self.translate(),
                    self.query(),
                    self.construct(),
                    self.simulate(),
                    self.validate(),
                    self.send_email(gmail_tools)
                ],
                manager_agent=self.brain_agent(),
                process=Process.hierarchical,
                verbose=True
            )