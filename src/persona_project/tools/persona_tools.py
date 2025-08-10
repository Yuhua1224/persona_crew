from crewai.tools import BaseTool
from pydantic import BaseModel, Field, validator
from typing import Type, List, Dict, Optional
import numpy as np
import pandas as pd
from collections import defaultdict, Counter


# =========================
# Tool 1: PersonaBuilderTool
# =========================

class PersonaBuilderInput(BaseModel):
    records: List[Dict] = Field(..., description="List of user records to aggregate")
    drop_fields: Optional[List[str]] = Field(
        default=["email"],
        description="Columns to exclude from aggregation/output"
    )
    categorical_modes: Optional[List[str]] = Field(
        default=["gender"],
        description="Categorical columns summarized by mode"
    )
    expected_fields: Optional[List[str]] = Field(
        default=None,
        description="If provided, final output will be filtered to this exact field list (e.g., the 59-field schema)"
    )
    round_ndigits: Optional[int] = Field(
        default=3,
        description="Round numeric means to this many decimal places"
    )

class PersonaBuilderTool(BaseTool):
    name: str = "Persona Builder"
    description: str = "Aggregates multiple user records into a 59-field representative profile: numeric means + categorical modes"
    args_schema: Type[BaseModel] = PersonaBuilderInput

    def _run(
        self,
        records: List[Dict],
        drop_fields: Optional[List[str]] = None,
        categorical_modes: Optional[List[str]] = None,
        expected_fields: Optional[List[str]] = None,
        round_ndigits: int = 3,
    ) -> Dict:
        if not records:
            return {"error": "No records provided.", "profile": {}, "sample_size": 0}

        df = pd.DataFrame(records)

        # Drop unwanted fields (e.g., email)
        if drop_fields:
            for col in drop_fields:
                if col in df.columns:
                    df = df.drop(columns=[col])

        # Numeric means
        numeric_df = df.select_dtypes(include=["number"])
        avg_profile = {}
        if not numeric_df.empty:
            means = numeric_df.mean(numeric_only=True)
            if round_ndigits is not None:
                means = means.round(round_ndigits)
            avg_profile.update(means.to_dict())

        # Categorical modes (e.g., gender)
        if categorical_modes:
            for col in categorical_modes:
                if col in df.columns:
                    mode_series = df[col].dropna().astype(str)
                    if not mode_series.empty:
                        counts = Counter(mode_series)
                        maxc = max(counts.values())
                        candidates = sorted([k for k, v in counts.items() if v == maxc])
                        avg_profile[col] = candidates[0]

        # Canonical schema filter
        if expected_fields:
            filtered = {}
            missing = []
            for f in expected_fields:
                if f in avg_profile:
                    filtered[f] = avg_profile[f]
                else:
                    missing.append(f)
            result = {
                "profile": filtered,
                "missing_fields": missing,
                "sample_size": int(len(records)),
            }
        else:
            result = {
                "profile": avg_profile,
                "sample_size": int(len(records)),
                "fields_included": sorted(list(avg_profile.keys())),
            }

        return result


# =========================
# Tool 2: PersonaAnalysisTool  (已依你的新規格改寫)
# =========================

class PersonaAnalysisInput(BaseModel):
    simulated_items: List[int] = Field(..., description="Newly simulated 50 answers (1-5, raw, not reverse-scored)")
    real_items_avg: List[float] = Field(..., description="Database per-item averages for the same 50 items (raw, not reverse-scored)")
    real_traits_avg: Dict[str, float] = Field(..., description="Database Big Five trait averages (already correctly scored)")
    questionnaire: List[Dict] = Field(..., description="List of 50 item metadata dicts with keys: id, trait, reverse?")

    @validator("simulated_items", "real_items_avg")
    def _len_50(cls, v):
        if len(v) != 50:
            raise ValueError("Expected 50 items.")
        return v

class PersonaAnalysisTool(BaseTool):
    name: str = "Persona Similarity Analyzer"
    description: str = (
        "Computes two comparisons: (1) Big Five trait similarity by converting the new 50-item responses into traits "
        "with reverse-scoring applied only for the simulated answers, compared against DB trait averages; "
        "(2) 50-item answer similarity grouped by trait (5 scores) plus one overall score, without reverse-scoring."
    )
    args_schema: Type[BaseModel] = PersonaAnalysisInput

    def _run(
        self,
        simulated_items: List[int],
        real_items_avg: List[float],
        real_traits_avg: Dict[str, float],
        questionnaire: List[Dict],
    ) -> Dict:
        if len(questionnaire) != 50:
            return {"error": "Questionnaire metadata must be 50 items."}

        # Helper: reverse for Likert 1-5
        def rev(x): return 6 - x

        # ---- A) 由「此次填答的 50 題」換算五大面向（這裡才用 reverse） ----
        trait_to_sim_values = defaultdict(list)
        for i, q in enumerate(questionnaire):
            trait = q.get("trait", "unknown")
            reverse = bool(q.get("reverse", False))
            val = int(simulated_items[i])
            val_scored = rev(val) if reverse else val
            trait_to_sim_values[trait].append(val_scored)

        simulated_traits = {}
        for trait, vals in trait_to_sim_values.items():
            simulated_traits[trait] = float(np.mean(vals)) if vals else float("nan")

        # ---- B) 五大人格相似度（逐面向）----
        # real_traits_avg 已是最終正確分數，這裡不再反向
        trait_similarity_pct = {}
        trait_pairs = {}
        for trait, sim_val in simulated_traits.items():
            real_val = real_traits_avg.get(trait)
            # 若 DB 沒這個 trait，略過
            if real_val is None:
                continue
            # 相似度：1 - abs差 / 4（都在 1~5 量尺）
            sim_score = 1.0 - abs(float(sim_val) - float(real_val)) / 4.0
            trait_similarity_pct[trait] = round(sim_score * 100.0, 2)
            trait_pairs[trait] = {
                "simulated_trait": round(float(sim_val), 3),
                "real_trait": round(float(real_val), 3)
            }

        # ---- C) 50 題填答相似度（不反向）：每個 trait 一個分數 + 總分 ----
        trait_item_diffs = defaultdict(list)
        all_abs_diffs = []
        for i, q in enumerate(questionnaire):
            trait = q.get("trait", "unknown")
            sim_raw = float(simulated_items[i])      # 不反向
            real_raw = float(real_items_avg[i])      # 不反向
            diff = abs(sim_raw - real_raw)
            trait_item_diffs[trait].append(diff)
            all_abs_diffs.append(diff)

        item_similarity_by_trait_pct = {}
        for trait, diffs in trait_item_diffs.items():
            if len(diffs) == 0:
                continue
            score = 1.0 - (float(np.mean(diffs)) / 4.0)
            item_similarity_by_trait_pct[trait] = round(score * 100.0, 2)

        overall_item_similarity_pct = round(
            (1.0 - float(np.mean(all_abs_diffs)) / 4.0) * 100.0, 2
        ) if all_abs_diffs else None

        # ---- 組合輸出 ----
        result = {
            # 五大人格：同時列出資料中的（real）與此次填答換算後（simulated），並給相似度
            "trait_scores": {
                "simulated": {k: round(v, 3) for k, v in simulated_traits.items()},
                "real": {k: round(float(v), 3) for k, v in real_traits_avg.items()},
            },
            "trait_similarity_pct": trait_similarity_pct,

            # 50 題填答相似度（不反向）：5 個分數（按 trait）+ 一個總分
            "item_similarity_pct": {
                "by_trait": item_similarity_by_trait_pct,
                "overall": overall_item_similarity_pct
            },

            # 參考資訊
            "diagnostics": {
                "traits_seen": sorted(set(list(trait_to_sim_values.keys()) + list(real_traits_avg.keys()))),
                "num_items": 50
            }
        }
        return result


# =========================
# Tool 3: QuestionnaireTool
# =========================

class QuestionnaireTraitInput(BaseModel):
    questionnaire: List[Dict] = Field(..., description="Full questionnaire metadata (list of dicts)")
    question_id: Optional[int] = Field(
        default=None,
        description="Single question ID to lookup"
    )
    question_ids: Optional[List[int]] = Field(
        default=None,
        description="Multiple question IDs to lookup"
    )
    include_text: Optional[bool] = Field(
        default=False,
        description="If true, include 'text' from metadata when available"
    )

class QuestionnaireTool(BaseTool):
    name: str = "Questionnaire Info Lookup"
    description: str = "Lookup trait/reverse (and optional text) for one or many question IDs"
    args_schema: Type[BaseModel] = QuestionnaireTraitInput

    def _run(
        self,
        questionnaire: List[Dict],
        question_id: Optional[int] = None,
        question_ids: Optional[List[int]] = None,
        include_text: bool = False,
    ) -> Dict:
        # Build index for O(1) lookups
        index = {}
        for q in questionnaire:
            qid = q.get("id")
            if qid is not None:
                index[int(qid)] = q

        def pack(qid: int) -> Dict:
            meta = index.get(int(qid))
            if not meta:
                return {"id": qid, "error": "Question ID not found"}
            out = {
                "id": qid,
                "trait": meta.get("trait"),
                "reverse": bool(meta.get("reverse", False)),
            }
            if include_text:
                out["text"] = meta.get("text")
            return out

        if question_id is not None:
            return pack(question_id)

        if question_ids:
            return {"items": [pack(qid) for qid in question_ids]}

        return {"error": "Provide question_id or question_ids."}
