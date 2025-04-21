from pydantic import BaseModel, Field
from typing import List, Literal

class LearningDataRequest(BaseModel):
    Online_Courses_Completed: List[int] = Field(alias="online_Courses_Completed")
    Participation_in_Discussions: List[Literal["Yes", "No"]] = Field(alias="participation_in_Discussions")
    Assignment_Completion_Rate: List[float] = Field(alias="assignment_Completion_Rate")
    Exam_Score: List[float] = Field(alias="exam_Score")

    class Config:
        allow_population_by_field_name = True