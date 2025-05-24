from pydantic import BaseModel, Field
from typing import List, Literal

class AssignmentGrade(BaseModel):
    doc_id: int
    grade: float

class SubjectAssignmentRate(BaseModel):
    subject_id: int
    subject_name: str
    assigment_grades: List[AssignmentGrade]
    avg_score: float

class SubjectTestRate(BaseModel):
    subject_id: int
    subject_name: str
    avg_score: float

class CoursePeriod(BaseModel):
    subject_id: int
    subject_name: str
# class LearningDataRequest(BaseModel):
#     Online_Courses_Completed: List[int] = Field(alias="online_Courses_Completed")
#     Participation_in_Discussions: List[Literal["Yes", "No"]] = Field(alias="participation_in_Discussions")
#     Assignment_Completion_Rate: List[float] = Field(alias="assignment_Completion_Rate")
#     Exam_Score: List[float] = Field(alias="exam_Score")
#
#     class Config:
#         allow_population_by_field_name = True

class LearningDataRequest(BaseModel):
    Online_Courses_Completed: List[int] = Field(alias="online_courses_completed")
    Participation_in_Discussions: List[Literal["Yes", "No"]] = Field(alias="participation_in_discussions")
    Assignment_Completion_Rate: List[SubjectAssignmentRate] = Field(alias="assignment_completion_rate")
    Exam_Score: List[SubjectTestRate] = Field(alias="exam_score")
    Course_period: List[CoursePeriod] = Field(alias="course_period")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True