#!/usr/bin/env python
import warnings
from persona_project.crew import PersonaCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    print("Persona Crew 啟動")
    persona_description = input("請輸入你想模擬的人格描述：").strip()

    if not persona_description:
        print("請輸入有效的人格描述。")
        return

    inputs = {
        "topic": persona_description
    }

    try:
        PersonaCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        print(f" 運行錯誤：{e}")


if __name__ == "__main__":
    run()
