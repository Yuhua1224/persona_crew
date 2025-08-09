from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Dict
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# --- Tool 1: PersonaBuilderTool ---
class PersonaBuilderInput(BaseModel):
    records: List[Dict] = Field(..., description="List of user records to average")

class PersonaBuilderTool(BaseTool):
    name: str = "Persona Builder"
    description: str = "Averages multiple user records into a representative persona profile"
    args_schema: Type[BaseModel] = PersonaBuilderInput

    def _run(self, records: List[Dict]) -> Dict:
        df = pd.DataFrame(records)
        numeric_df = df.select_dtypes(include=["number"])
        avg_profile = numeric_df.mean().to_dict()

        # Special handling for gender: use mode
        if "gender" in df.columns:
            gender_mode = df["gender"].mode(dropna=True)
            if not gender_mode.empty:
                avg_profile["gender"] = gender_mode.iloc[0]

        return avg_profile

# --- Tool 2: PersonaAnalysisTool ---
class PersonaAnalysisInput(BaseModel):
    simulated: List[int] = Field(..., description="List of 50 simulated answers")
    real_avg: List[float] = Field(..., description="List of 50 real average answers")
    questionnaire: List[Dict] = Field(..., description="Questionnaire metadata including trait and reverse")

class PersonaAnalysisTool(BaseTool):
    name: str = "Persona Similarity Analyzer"
    description: str = "Compares two sets of 50 Big-Five responses and returns trait-level similarity"
    args_schema: Type[BaseModel] = PersonaAnalysisInput

    def _run(self, simulated: List[int], real_avg: List[float], questionnaire: List[Dict]) -> Dict:
        trait_map = defaultdict(lambda: {"sim": [], "real": []})

        for i, q in enumerate(questionnaire):
            trait = q["trait"]
            reverse = q.get("reverse", False)

            sim = 6 - simulated[i] if reverse else simulated[i]
            real = 6 - real_avg[i] if reverse else real_avg[i]

            trait_map[trait]["sim"].append(sim)
            trait_map[trait]["real"].append(real)

        similarity = {}
        for trait, values in trait_map.items():
            sim_vec = np.array(values["sim"])
            real_vec = np.array(values["real"])
            sim_score = 1 - np.mean(np.abs(sim_vec - real_vec)) / 4  # normalized similarity
            similarity[trait] = round(sim_score * 100, 2)  # percentage

        return similarity

# --- Tool 3: QuestionnaireTool ---
class QuestionnaireTraitInput(BaseModel):
    question_id: int = Field(..., description="ID of the question")
    questionnaire: List[Dict] = Field(..., description="Full questionnaire metadata")

class QuestionnaireTool(BaseTool):
    name: str = "Questionnaire Info Lookup"
    description: str = "Given a question ID, return its associated trait and reverse status"
    args_schema: Type[BaseModel] = QuestionnaireTraitInput

    def _run(self, question_id: int, questionnaire: List[Dict]) -> Dict:
        for q in questionnaire:
            if q["id"] == question_id:
                return {
                    "trait": q.get("trait"),
                    "reverse": q.get("reverse", False)
                }
        return {"error": f"Question ID {question_id} not found."}
