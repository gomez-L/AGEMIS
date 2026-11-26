from pydantic import BaseModel, Field
from typing import Literal


class TransmissionStep(BaseModel):
    stage: Literal["exposure", "initial_impact", "response", "amplification",
                   "systemic_consequence", "containment"]
    mechanism: str = Field(min_length=1, description="Actors, exposure and mechanism. State missing information explicitly")
    basis: Literal["evidence", "simulation", "assumption"]
    references: list[str] = Field(default_factory=list, description="Only supplied evidence IDs s1, s2, ... or event IDs e1, e2, ...")


class MonitoringIndicator(BaseModel):
    indicator: str = Field(min_length=1)
    interpretation: str = Field(min_length=1, description="Which mechanism a change would support or weaken")


class ScenarioNarration(BaseModel):
    """The scenario is the central-bank briefing - the spupporting  detail stays separate."""
    title: str = Field(min_length=1, max_length=100, description="Concrete plain-language headline, at most 10 words")
    narration: str = Field(min_length=1, max_length=2200, description="Three short paragraphs: development, financial consequences, escalation and containment. 140-200 words, no reference codes or model terminology")
    counterfactual_assumption: str = Field(min_length=1, max_length=260, description="One natural What if question about a hypothetical real-world development")
    watch_for: list[MonitoringIndicator] = Field(min_length=1, max_length=3)
    key_uncertainty: str = Field(min_length=1, max_length=300, description="The key missing fact or condition, in one plain-language sentence")
    analyst_question: str = Field(min_length=1, max_length=240, description="One specific, practical question for follow-up")
    supporting_analysis: list[TransmissionStep] = Field(min_length=1, max_length=4, description="Only the key claims behind the scenario; distinguish evidence, simulation and assumptions")
    trigger_signal: str = Field(default="", description="Short supporting note about the originating evidence")
    novelty_element: str = Field(default="", description="Short supporting note about what changes")
