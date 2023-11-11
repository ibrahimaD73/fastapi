
from pydantic import BaseModel

class SummarizeRequestModel(BaseModel):
    text: str


class ExtractSkillsRequestModel(BaseModel):
    text: str

class MatchJobCandidateRequestModel(BaseModel):
    job_description: str
    candidate_resume: str

