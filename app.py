import os
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pre_proccessing.InputHandle import InputHandle
from pydantic import ValidationError
from flask_cors import CORS
import requests
from response.ApiRespone import ApiResponse
from request.LearningDataRequest import LearningDataRequest
from request.NewDataRequest import MultipleNewDataRequest
from model.RecommendationSystem import DocumentRecommendationSystem
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

CORS(
    app,
    resources={r"/api/*": {"origins": ["http://localhost:5173", "http://localhost:8088", "http://localhost:8084", "http://dtuforyou.xyz", "https://dtuforyou.xyz"]}},
    methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=True,
)

model= joblib.load("./model/model_v1_2.pkl")
scaler= joblib.load("./model/scaler_v1_2.pkl")
le= joblib.load("./model/labelEncoder_v1_2.pkl")
input_handler = InputHandle(scaler)
folder_path="./dataset/"

rs= DocumentRecommendationSystem()
documents_df=pd.read_csv(f"{folder_path}documents.csv")
ratings_df=pd.read_csv(f"{folder_path}ratings.csv")
users_df = pd.read_csv(f"{folder_path}users.csv")
rs.load_data(ratings_df, documents_df,users_df['account_id'].unique().tolist())
rs.train_collaborative_filtering()
chatbot_url= os.getenv("CHATBOT_URL")

BASE_URL="/api/v1/recommendation"

@app.route(f'{BASE_URL}/get-solution', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        learning_request= LearningDataRequest(**data)
        dataDict = input_handler.handle_data_input(assignment_rate=learning_request.Assignment_Completion_Rate,
                                                   exam_rate=learning_request.Exam_Score)

        print("data dict: ", dataDict)

        df_dict = {}

        for k,v in vars(learning_request).items():
            if k == "Assignment_Completion_Rate":
                value = dataDict["avg_assign_score"]
                df_dict[k] = value
            elif k == "Exam_Score":
                value = dataDict["avg_exam_score"]
                df_dict[k] = value
            else:
                if k != "Course_period":
                    df_dict[k] = v[0]

        df_input = pd.DataFrame([df_dict])

        print("Converted DataFrame:", df_input)

        # Rename cột nếu tồn tại
        rename_map = {
            "Assignment_Completion_Rate": "Assignment_Completion_Rate (%)",
            "Exam_Score": "Exam_Score (%)"
        }
        df_input = df_input.rename(columns={k: v for k, v in rename_map.items() if k in df_input.columns})

        print("Renamed columns:", df_input.columns.tolist())
        print("Converted DataFrame:", df_input)

        # Handle data pre-processing
        df_processed = InputHandle.data_pre_processing(df_input)
        print("Processed DataFrame:", df_processed)
        copy_df = df_processed.copy()

        # Handle feature scaling with method standardscaler
        df_feature_scale= input_handler.input_feature_scaling(df_processed)
        print("Feature scale DataFrame:", df_feature_scale)

        # Predict with model
        result_prediction = model.predict(df_feature_scale)
        print("Result prediction:", result_prediction)


        result= le.inverse_transform(result_prediction)
        extract_student_low_scores =[]
        score_student= result.tolist()
        for index,score in enumerate(score_student):
            if score in ["C","D"] :
                extract_student_low_scores.append({
                    "index": index,
                    "score": score
                })

        filter_student_df = {
                **copy_df.to_dict(),
                "Final Grade": score_student[0],
                "subject_weakens": dataDict,
                "courses_period": [c.model_dump() for c in learning_request.Course_period]
        }
        if score_student[0] not in ["C","D"] :
            return ApiResponse.success(data="Bạn đã có sự cải thiện rõ rệt trong việc học tập. Việc chăm chỉ làm bài tập và hoàn thành các bài quiz đều đặn đang giúp bạn tiến bộ từng ngày. Cố gắng phát huy nhé!")

        print(filter_student_df)

        uri= chatbot_url+"/get-solutions"

        response= requests.post(uri, json=filter_student_df)

        return jsonify(response.json())
        # return ApiResponse.success(message="Recommend account success",code=200,data=response)
    except ValidationError as e:
        return ApiResponse.error(message="Invalid data format",code=400)
    except Exception as e:
        return ApiResponse.error(message=str(e))

@app.route(f'{BASE_URL}/document/get-by-user', methods=['GET'])
def handle_recommend_for_user():
    try:
        account_id= request.args.get('account_id')
        if not account_id:
            return jsonify({"error": "No input data provided"}), 400
        account_id_converter= f"USER_{int(account_id):05d}"
        recommended_docs= rs.get_collaborative_filtering_recommendations(account_id_converter)
        return ApiResponse.success(message="Recommend document success",code=200,data=recommended_docs)

    except ValidationError as e:
        return ApiResponse.error(message="Invalid data format",code=400)
    except Exception as e:
        return ApiResponse.error(e,code=500)

@app.route(f'{BASE_URL}/document/collect-data', methods=['POST'])
def handle_collect_data():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        new_data_request = MultipleNewDataRequest(**data)
        data_converter = input_handler.handle_data_request(new_data_request)
        rs.update_model(new_documents=data_converter.newDocuments,
                        update_document=data_converter.updateDocuments,
                        new_interactions=data_converter.userInteractions,
                        new_users=data_converter.userRequests,
                        folder_path=folder_path)
        return ApiResponse.success(message="Multiple new data added!", code=200)
    except Exception as e:
        return ApiResponse.error(e,code=500)

@app.route(f'{BASE_URL}/load-data', methods=['POST'])
def load_data_to_csv():
    try:
        data= request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        new_data_request = MultipleNewDataRequest(**data)
        data_converter= input_handler.handle_data_request(new_data_request)
        documents_df1 = pd.read_csv(f"{folder_path}/documents.csv")
        new_docs_df = pd.DataFrame(data_converter.newDocuments,
                                   columns=['document_id', 'title', 'category', 'content', 'popularity'])
        updated_documents_df = pd.concat([documents_df1, new_docs_df]).drop_duplicates(
            subset='document_id').reset_index(drop=True)
        updated_documents_df.to_csv(f"{folder_path}/documents.csv", index=False)
        # new_docs_df.to_csv(f"{folder_path}/documents.csv", index=False)
        return ApiResponse.success(message="Multiple new data added!", code=200)
    except Exception as e:
        return ApiResponse.error(e,code=500)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
