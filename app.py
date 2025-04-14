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

app = Flask(__name__)

CORS(
    app,
    resources={r"/api/*": {"origins": ["http://localhost:5173", "http://localhost:8088", "http://localhost:8084"]}},
    methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=True,
)

model= joblib.load("./model/model.pkl")
scaler= joblib.load("./model/scaler.pkl")
le= joblib.load("./model/labelEncoder.pkl")
input_handler = InputHandle(scaler)
folder_path="./dataset/"

rs= DocumentRecommendationSystem()
documents_df=pd.read_csv(f"{folder_path}documents.csv")
ratings_df=pd.read_csv(f"{folder_path}ratings.csv")
rs.load_data(ratings_df, documents_df)
rs.train_collaborative_filtering()


@app.route('/api/v1/recommendation/get-solution', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        learning_request= LearningDataRequest(**data)
        # create dataframe from data after validation success
        df_input = pd.DataFrame(learning_request)
        print("Converted DataFrame:", df_input)

        # rename column
        df_input = df_input.rename(columns={
            "Assignment_Completion_Rate": "Assignment_Completion_Rate (%)",
            "Exam_Score": "Exam_Score (%)",
            "Time_Spent_on_Social_Media": "Time_Spent_on_Social_Media (hours/week)"
        })

        print(df_input.columns)

        # Handle data pre-processing
        df_processed = InputHandle.data_pre_processing(df_input)
        print("Processed DataFrame:", df_processed)

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
                **df_input.to_dict(),
                "Final Grade": score_student[0]
        }
        if score_student[0] not in ["C","D"] :
            return ApiResponse.success(data="Tu es excellent")

        print(filter_student_df)

        response= requests.post("http://127.0.0.1:5002/api/v1/get-solutions", json=filter_student_df)

        return jsonify(response.json())

    except ValidationError as e:
        return ApiResponse.error(message="Invalid data format",code=400)
    except Exception as e:
        return ApiResponse.error(message=str(e))

@app.route('/api/v1/recommendation/document/get-by-user', methods=['GET'])
def handle_recommend_for_user():
    try:
        account_id= request.args.get('account_id')
        if not account_id:
            return jsonify({"error": "No input data provided"}), 400

        recommended_docs= rs.get_collaborative_filtering_recommendations(account_id)
        return ApiResponse.success(message="Multiple ratings added!",code=200,data=recommended_docs)

    except ValidationError as e:
        return ApiResponse.error(message="Invalid data format",code=400)
    except Exception as e:
        return ApiResponse.error(e,code=500)

def handle_data_request(data:MultipleNewDataRequest):
    fixed_new_documents = []
    fixed_updated_documents = []
    fixed_user_interaction= []
    fixed_user_requests= []
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
    data.updateDocuments= fixed_updated_documents
    data.newDocuments=fixed_new_documents
    data.userInteractions=fixed_user_interaction
    data.userRequests=fixed_user_requests
    return data

@app.route('/api/v1/recommendation/document/collect-data', methods=['POST'])
def handle_collect_data():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        new_data_request = MultipleNewDataRequest(**data)
        data_converter = handle_data_request(new_data_request)
        rs.update_model(new_documents=data_converter.newDocuments,
                        update_document=data_converter.updateDocuments,
                        new_interactions=data_converter.userInteractions,
                        new_users=data_converter.userRequests,
                        folder_path=folder_path)
        return ApiResponse.success(message="Multiple new data added!", code=200)
    except Exception as e:
        return ApiResponse.error(e,code=500)

@app.route('/api/v1/recommendation/load-data', methods=['POST'])
def load_data_to_csv():
    try:
        data= request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        new_data_request = MultipleNewDataRequest(**data)
        data_converter= handle_data_request(new_data_request)
        # rs.update_model(new_documents=data_converter.newDocuments,
        #                 update_document=data_converter.updateDocuments,
        #                 new_interactions=data_converter.userInteractions,
        #                 new_users=data_converter.userRequests,
        #                 folder_path=folder_path)
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
