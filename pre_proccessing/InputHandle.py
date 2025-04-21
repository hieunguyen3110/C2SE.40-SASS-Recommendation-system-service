from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

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