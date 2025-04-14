from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

class InputHandle:
    def __init__(self, scaler):
        self.scaler = scaler

    @staticmethod
    def data_pre_processing(df_input):
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore',
                                categories=[["Auditory", "Kinesthetic", "Reading/Writing", "Visual"]])
        le = LabelEncoder()

        # Fit và transform dữ liệu Preferred_Learning_Style
        encoded = encoder.fit_transform(df_input[["Preferred_Learning_Style"]])

        # Lấy tên cột từ OneHotEncoder sau khi fit
        column_names = encoder.get_feature_names_out(["Preferred_Learning_Style"])

        # Tạo DataFrame từ dữ liệu đã mã hóa
        df_encoded = pd.DataFrame(encoded, columns=column_names)

        # Loại bỏ cột cũ và ghép cột đã mã hóa
        df_input = df_input.drop(columns=["Preferred_Learning_Style"]).reset_index(drop=True)
        df_encoded = df_encoded.reset_index(drop=True)  # Reset index để tránh lỗi khi concat
        df_input = pd.concat([df_input, df_encoded], axis=1)

        # Mã hóa Participation_in_Discussions
        df_input["Participation_in_Discussions"] = le.fit_transform(df_input["Participation_in_Discussions"])
        return df_input

    def input_feature_scaling(self,df_input):
        # Xác định các cột cần scale
        numerical_columns = [
            "Study_Hours_per_Week",
            "Online_Courses_Completed",
            "Assignment_Completion_Rate (%)",
            "Exam_Score (%)",
            "Time_Spent_on_Social_Media (hours/week)",
            "Sleep_Hours_per_Night"
        ]

        # Chuyển đổi kiểu dữ liệu trước khi scale
        df_input[numerical_columns] = df_input[numerical_columns].astype(float)

        # Scale các cột số
        df_input[numerical_columns] = self.scaler.transform(df_input[numerical_columns])

        return df_input