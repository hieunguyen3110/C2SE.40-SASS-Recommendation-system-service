from pydantic import BaseModel
from typing import List, Literal

class LearningDataRequest(BaseModel):
    Study_Hours_per_Week: List[int]
    Preferred_Learning_Style: List[Literal["Auditory", "Kinesthetic", "Reading/Writing", "Visual"]]
    Online_Courses_Completed: List[int]
    Participation_in_Discussions: List[Literal["Yes", "No"]]
    Assignment_Completion_Rate: List[float]  # Đổi tên bỏ ký tự `%`
    Exam_Score: List[float]  # Đổi tên bỏ ký tự `%`
    Time_Spent_on_Social_Media: List[float]  # Đơn vị: giờ/tuần
    Sleep_Hours_per_Night: List[float]  # Đơn vị: giờ/đêm