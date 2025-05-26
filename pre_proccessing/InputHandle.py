from typing import List, Dict, Any

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

from request.LearningDataRequest import SubjectAssignmentRate, SubjectTestRate, CoursePeriod
from request.NewDataRequest import MultipleNewDataRequest
from response.ApiRespone import ApiResponse


class InputHandle:
    def __init__(self, scaler):
        self.scaler = scaler

    @staticmethod
    def data_pre_processing(df_input):
        try:
            le = LabelEncoder()
            # Mã hóa Participation_in_Discussions
            df_input["Participation_in_Discussions"] = le.fit_transform(df_input["Participation_in_Discussions"])
            return df_input

        except Exception as e:
            print("Exception: ", e)
            return ApiResponse.error(message="Error when try pre-process data", code=400)

    def input_feature_scaling(self,df_input):
        # Xác định các cột cần scale
        numerical_columns = [
            "Online_Courses_Completed",
            "Assignment_Completion_Rate (%)",
            "Exam_Score (%)"
        ]

        # Chuyển đổi kiểu dữ liệu trước khi scale
        df_input[numerical_columns] = df_input[numerical_columns].astype(float)

        # Scale các cột số
        df_input[numerical_columns] = self.scaler.transform(df_input[numerical_columns])

        return df_input

    @staticmethod
    def handle_data_request(data: MultipleNewDataRequest):
        fixed_new_documents = []
        fixed_updated_documents = []
        fixed_user_interaction = []
        fixed_user_requests = []
        if data.newDocuments:
            for doc in data.newDocuments:
                fixed_new_documents.append({
                    "document_id": f"DOC_{int(doc.document_id):05d}",
                    "title": doc.title,
                    "category": doc.category,
                    "content": doc.content,
                    "popularity": doc.popularity
                })
        if data.updateDocuments:
            for doc in data.updateDocuments:
                fixed_updated_documents.append({
                    "document_id": f"DOC_{int(doc.document_id):05d}",
                    "title": doc.title,
                    "category": doc.category,
                    "content": doc.content,
                    "popularity": doc.popularity
                })
        if data.userInteractions:
            for userInteraction in data.userInteractions:
                fixed_user_interaction.append({
                    "account_id": f"USER_{int(userInteraction.account_id):05d}",
                    "document_id": f"DOC_{int(userInteraction.document_id):05d}",
                    "timestamp": userInteraction.timestamp,
                    "rating": userInteraction.rating,
                    "view_time": userInteraction.view_time
                })
        if data.userRequests:
            for userRequest in data.userRequests:
                fixed_user_requests.append({
                    "account_id": f"USER_{int(userRequest.account_id):05d}",
                })
        data.updateDocuments = fixed_updated_documents
        data.newDocuments = fixed_new_documents
        data.userInteractions = fixed_user_interaction
        data.userRequests = fixed_user_requests
        return data

    @staticmethod
    def handle_data_input(assignment_rate: List[SubjectAssignmentRate], exam_rate: List[SubjectTestRate]):
        try:
            assignment_score_rate = 0
            exam_score_rate = 0
            assignment_count = 0
            exam_count = 0
            subject_weaken_dict: Dict[int, Dict[str, Any]] = {}
            for assignment in assignment_rate:
                assignment_count+=1
                assignment_score_rate+=assignment.avg_score
                if assignment.avg_score <= 65:
                    subject_weaken_dict[assignment.subject_id] = {
                        "subject_id": assignment.subject_id,
                        "subject_name": assignment.subject_name,
                        "avg_score": assignment.avg_score,
                    }

            for exam in exam_rate:
                exam_count+=1
                exam_score_rate+=exam.avg_score
                if exam.avg_score <= 65 and exam.subject_id not in subject_weaken_dict:
                    subject_weaken_dict[exam.subject_id] = {
                        "subject_id": exam.subject_id,
                        "subject_name": exam.subject_name,
                        "avg_score": exam.avg_score,
                    }
            avg_assign_score = 0
            avg_exam_score = 0
            if assignment_count != 0:
                avg_assign_score = assignment_score_rate / assignment_count

            if exam_count != 0:
                avg_exam_score = exam_score_rate / exam_count

            assignment_subject_ids = set()
            for assignment in assignment_rate:
                assignment_subject_ids.add(assignment.subject_id)
                if assignment.subject_id in subject_weaken_dict:
                    subject_weaken_dict[assignment.subject_id]["assigment_grades"] = [grade.model_dump() for grade in assignment.assigment_grades]

            for subject_id in subject_weaken_dict:
                if subject_id not in assignment_subject_ids:
                    subject_weaken_dict[subject_id]["assigment_grades"] = []



            return {
                "avg_assign_score": avg_assign_score,
                "avg_exam_score": avg_exam_score,
                "subject_weaken_dict": subject_weaken_dict
            }
        except Exception as e:
            print("Exception: ", e)