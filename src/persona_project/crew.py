import os
import yaml
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from persona_project.tools.persona_tools import (
    PersonaBuilderTool,
    PersonaAnalysisTool,
    QuestionnaireTool
)
from crewai_tools import MySQLSearchTool, MCPServerAdapter


# --- Load Big Five questionnaire metadata ---
def load_questionnaire(path="data/questionnaire.yml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

questionnaire = load_questionnaire()

# 提供給「simulate」的淨化版問卷：只含 id 與 text（不含 trait/reverse）
simulation_questions = [
    {"id": q.get("id"), "text": q.get("text", "")} for q in questionnaire
]

# --- Tools ---
mysql_tool = MySQLSearchTool(
    db_uri=os.getenv("PERSONA_MYSQL_URI"),
    table_name=os.getenv("PERSONA_SURVEY_TABLE", "survey_answers"),
)

persona_builder_tool = PersonaBuilderTool()     # 給 database_agent 在 query 任務做「聚合→59欄位 profile」
persona_analysis_tool = PersonaAnalysisTool()  # 給 validation_agent 比對五大面向與50題相似度
questionnaire_tool = QuestionnaireTool()       # 給 simulator / validation 查題目 trait/reverse


@CrewBase
class PersonaCrew:
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"
    mcp_server_params = {
        "url": os.getenv("GMAIL_MCP_ENDPOINT"),
        "transport": "sse"
    }

    # ============= Agents =============
    @agent
    def brain_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["brain_agent"],
            tools=[], allow_delegation=True, verbose=True
        )

    @agent
    def translation_agent(self) -> Agent:
        # 把自由文字 → 可下DB的硬性條件（含 Qn > / < 題目條件）
        return Agent(
            config=self.agents_config["translation_agent"],
            tools=[questionnaire_tool],  # 若要輔助理解題目編號，可用
            verbose=True
        )

    @agent
    def database_agent(self) -> Agent:
        # 先用 MySQLSearchTool 依 translate 的條件撈資料，
        # 再用 PersonaBuilderTool 聚合成 59 欄位 profile（50題平均 + 5 traits + 人口統計；gender 取眾數）
        return Agent(
            config=self.agents_config["database_agent"],
            tools=[mysql_tool, persona_builder_tool],
            verbose=True
        )

    @agent
    def persona_construction_agent(self) -> Agent:
        # 只把 59 欄位 profile 轉成「高可讀 persona 描述」
        return Agent(
            config=self.agents_config["persona_construction_agent"],
            tools=[], verbose=True
        )

    @agent
    def simulator_agent(self) -> Agent:
        # 依 persona 描述 + 問卷中繼資料填 50 題
        return Agent(
            config=self.agents_config["persona_simulator_agent"],
            verbose=True
        )

    @agent
    def validation_agent(self) -> Agent:
        # 用新版 PersonaAnalysisTool：
        # 反向題只用在「把這次 50 題換算成 Big Five」；
        # 再與 DB 的 Big Five（已正確計分）相比 → 特質相似度
        # 題目相似度：按 trait 給 5 個分數 + 1 個總分（不反向）
        return Agent(
            config=self.agents_config["validation_agent"],
            tools=[persona_analysis_tool, questionnaire_tool],
            verbose=True
        )

    @agent
    def email_agent(self, tools) -> Agent:
        return Agent(
            config=self.agents_config["email_agent"],
            tools=tools, verbose=True
        )

    # ============= Tasks =============
    @task
    def translate(self) -> Task:
        # 輸出：可下 DB 的條件 dict（含年齡/性別/Big Five 門檻/單題條件 Qn）
        return Task(
            config=self.tasks_config["translate"],
            agent=self.translation_agent()
        )

    @task
    def query(self) -> Task:
        # 輸出：59 欄位代表 profile（及必要的項目平均）
        # 步驟：translation 條件 → MySQL 查詢 → PersonaBuilder 聚合
        return Task(
            config=self.tasks_config["query"],
            context=[self.translate()],
            agent=self.database_agent()
        )

    @task
    def construct(self) -> Task:
        # 輸出：高可讀 persona 描述（多段文字）
        # 輸入：上一個任務產生的 59 欄位 profile
        return Task(
            config=self.tasks_config["construct"],
            context=[self.query()],
            agent=self.persona_construction_agent()
        )
    @task
    def simulate(self) -> Task:
        # 只把「題目文字/題號」給 Simulator；不給 trait/reverse
        return Task(
            config=self.tasks_config["simulate"],
            context=[self.construct()],
            inputs={"questions": simulation_questions},  # ← 換成只含題目文字的版本
            agent=self.simulator_agent()
        )


    @task
    def validate(self) -> Task:
        # 輸出：兩組相似度
        # 1) 五大面向相似度（列出 simulated vs real trait 分數，並給相似度）
        # 2) 50 題相似度（分 trait 的5個分數 + 1個總分，不反向）
        # 輸入來源：
        #   - simulate() → simulated_items（50題）
        #   - query()    → real_items_avg（每題平均）、real_traits_avg（已正確計分的五大面向）
        return Task(
            config=self.tasks_config["validate"],
            context=[self.simulate(), self.query()],
            inputs={"questionnaire": questionnaire},
            agent=self.validation_agent()
        )

    @task
    def send_email(self, tools) -> Task:
        return Task(
            config=self.tasks_config["send_email"],
            context=[self.validate()],
            agent=self.email_agent(tools),
            markdown=True
        )

    # ============= Crew =============
    @crew
    def crew(self) -> Crew:
        with MCPServerAdapter(self.mcp_server_params) as mcp_tools:  
            gmail_tools = [t for t in mcp_tools if t.name == "gmail_send"] # gmail_tools 就是從 MCP 拿到的工具
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
