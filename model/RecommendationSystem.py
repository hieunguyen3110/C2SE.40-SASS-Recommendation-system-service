import pandas as pd
import logging

from response.ApiRespone import ApiResponse

logging.basicConfig(level=logging.WARNING)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lightfm import LightFM
from lightfm.data import Dataset as LFMDataset
from lightfm.evaluation import precision_at_k, recall_at_k

class DocumentRecommendationSystem:
    def __init__(self):
        self.num_item_features = None
        self.reverse_user_mapping = None
        self.item_feature_mapping = None
        self.user_interactions = None
        self.document_data = None
        self.all_user_ids= None
        self.content_similarity_matrix = None
        self.lightfm_model = LightFM(loss='warp', learning_rate=0.05, no_components=30)
        self.lightfm_dataset = None
        self.user_mapping = None
        self.item_mapping = None
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=2,
            max_features=5000
        )

    def load_data(self, user_interactions_data, document_data,user_ids=None):
        """
        Nạp dữ liệu người dùng và tài liệu
        """
        self.user_interactions = user_interactions_data
        self.document_data = document_data
        self.user_interactions['rating'] = self.user_interactions.apply(
            lambda row: min(5, max(1, row['view_time'] / 60 * 5)) if pd.isna(row['rating']) else row['rating'], axis=1
        )
        if user_ids is not None:
            self.all_user_ids = user_ids


    def train_collaborative_filtering(self):
        """
        Huấn luyện mô hình collaborative filtering với LightFM
        """
        # Chuẩn bị dữ liệu cho LightFM
        interactions_df = self.user_interactions.dropna(subset=['rating']).copy()

        # Khởi tạo dataset LightFM
        self.lightfm_dataset = LFMDataset()

        self.num_item_features = len(self.document_data['category'].unique())

        # Thêm thông tin users và items
        self.lightfm_dataset.fit(
            users=self.all_user_ids,
            items=self.document_data['document_id'].unique(),
            item_features=self.document_data['category'].unique()  # Sử dụng category làm feature
        )
        # Tạo ánh xạ user/item
        self.user_mapping, self.reverse_user_mapping, self.item_mapping, self.item_feature_mapping = self.lightfm_dataset.mapping()

        # Tạo ma trận tương tác
        user_item_matrix = self.lightfm_dataset.build_interactions(
            [(row['account_id'], row['document_id'], row['rating'])
             for _, row in interactions_df.iterrows()]
        )[0]

        # Tạo ma trận item features
        item_features = self.lightfm_dataset.build_item_features(
            [(doc_id, [self.document_data[self.document_data['document_id'] == doc_id]['category'].iloc[0]])
             for doc_id in self.document_data['document_id']]
        )
        print(user_item_matrix.shape)

        # Huấn luyện mô hình
        self.lightfm_model.fit(
            user_item_matrix,
            item_features=item_features,
            epochs=30,
            num_threads=4
        )
        if user_item_matrix.shape[0] > 1:
            train_precision = precision_at_k(
                self.lightfm_model,
                user_item_matrix,
                item_features=item_features,
                k=10
            ).mean()

            train_recall = recall_at_k(
                self.lightfm_model,
                user_item_matrix,
                item_features=item_features,
                k=10
            ).mean()
        else:
            train_precision = 0.0
            train_recall = 0.0

        return {'precision': train_precision, 'recall': train_recall}

    def get_collaborative_filtering_recommendations(self, user_id, top_n=6):
        """
        Trả về top N tài liệu được đề xuất dựa trên collaborative filtering
        """
        # Kiểm tra xem user_id có trong mapping không
        if user_id not in self.user_mapping:
            # Nếu không có, trả về tài liệu phổ biến
            return self.document_data.sort_values('popularity', ascending=False)['document_id'].head(top_n).tolist()

        # Lấy internal user_id
        internal_user_id = self.user_mapping[user_id]

        # Lấy item features để dự đoán
        item_features = self.lightfm_dataset.build_item_features(
            [(doc_id, [self.document_data[self.document_data['document_id'] == doc_id]['category'].iloc[0]])
             for doc_id in self.document_data['document_id']]
        )

        # Lấy internal item_ids
        internal_item_ids = np.array(list(self.item_mapping.values()))

        # Dự đoán điểm số cho tất cả tài liệu
        scores = self.lightfm_model.predict(
            internal_user_id,
            internal_item_ids,
            item_features=item_features
        )

        # Lấy chỉ mục của top N tài liệu được đề xuất
        top_indices = np.argsort(-scores)[:top_n]
        top_internal_item_ids = internal_item_ids[top_indices]

        # Chuyển đổi từ internal item_ids về document_ids
        reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
        top_document_ids = [reverse_item_mapping[item_id] for item_id in top_internal_item_ids]

        return top_document_ids

    @staticmethod
    def update_csv_data(folder_path, new_docs_df, new_interactions_df, update_docs_df=None,new_user_ids=None):
        try:
            # Đọc file cũ
            documents_df = pd.read_csv(f"{folder_path}/documents.csv")
            ratings_df = pd.read_csv(f"{folder_path}/ratings.csv")
            users_df = pd.read_csv(f"{folder_path}/users.csv")

            # ✅ Cập nhật các document đã tồn tại nếu có (từ update_docs_df)
            if update_docs_df is not None and not update_docs_df.empty:
                for _, row in update_docs_df.iterrows():
                    doc_id = row['document_id']
                    if doc_id in documents_df['document_id'].values:
                        for col in ['title', 'category', 'content', 'popularity']:
                            if col in row and pd.notna(row[col]):
                                documents_df.loc[documents_df['document_id'] == doc_id, col] = row[col]

            # Nối dữ liệu mới vào DataFrame cũ
            updated_documents_df = pd.concat([documents_df, new_docs_df]).drop_duplicates(
                subset='document_id').reset_index(drop=True)
            updated_ratings_df = pd.concat([ratings_df, new_interactions_df]).drop_duplicates(
                subset=['account_id', 'document_id']).reset_index(drop=True)

            if new_user_ids:
                existing_user_ids = users_df['account_id'].unique()
                truly_new_user_ids = [uid for uid in new_user_ids if uid not in existing_user_ids]
                if truly_new_user_ids:
                    new_users_df = pd.DataFrame({'account_id': truly_new_user_ids})
                    users_df = pd.concat([users_df, new_users_df]).drop_duplicates(subset='account_id').reset_index(
                        drop=True)

            # Ghi đè vào file CSV để lưu dữ liệu đã cập nhật
            updated_documents_df.to_csv(f"{folder_path}/documents.csv", index=False)
            updated_ratings_df.to_csv(f"{folder_path}/ratings.csv", index=False)
            users_df.to_csv(f"{folder_path}/users.csv", index=False)

            print("Dữ liệu đã được thêm và cập nhật thành công vào file CSV.")

        except Exception as e:
            print("Lỗi khi cập nhật file CSV:", e)


    def update_model(self, new_documents,update_document, new_interactions, new_users,folder_path):
        """
        Cập nhật mô hình LightFM với dữ liệu người dùng và tài liệu mới
        """
        # Cập nhật document_data với tài liệu mới
        try:
            fixed_new_documents = [dict(doc) for doc in new_documents]
            new_docs_df = pd.DataFrame(fixed_new_documents,columns=['document_id', 'title', 'category', 'content', 'popularity'])
            new_docs_df = new_docs_df.reindex(columns=self.document_data.columns, fill_value=None)
            self.document_data = pd.concat([self.document_data, new_docs_df]).drop_duplicates(subset='document_id').reset_index(drop=True)
            #update document
            for doc in update_document:
                doc_id = doc["document_id"]
                if doc_id in self.document_data['document_id'].values:
                    self.document_data.loc[self.document_data['document_id'] == doc_id, 'popularity'] = doc["popularity"]

            update_docs_df = pd.DataFrame([dict(doc) for doc in update_document],
                                          columns=['document_id', 'title', 'category', 'content', 'popularity'])

            # Cập nhật user_mapping và interactions nếu có user mới
            fixed_new_interaction= [dict(interaction) for interaction in new_interactions]
            new_interactions_df = pd.DataFrame(fixed_new_interaction,columns=['account_id', 'document_id', 'timestamp', 'view_time', 'rating'])
            new_interactions_df = new_interactions_df.reindex(columns=self.user_interactions.columns, fill_value=None)
            new_interactions_df['rating'] = new_interactions_df.apply(
                lambda row: min(5, max(1, row['view_time'] / 60 * 5)) if pd.isna(row['rating']) else row['rating'],
                axis=1
            )
            self.user_interactions = pd.concat([self.user_interactions, new_interactions_df]).reset_index(drop=True)

            new_user_ids = [user["account_id"] for user in new_users if user["account_id"] not in self.user_mapping]

            # 🔍 Lấy thêm các account_id đã có trong interactions nhưng chưa có trong user_mapping
            interaction_user_ids = self.user_interactions['account_id'].unique()
            missing_user_ids = [uid for uid in interaction_user_ids if uid not in self.user_mapping]

            if missing_user_ids:
                print(f"⚠️ Missing {len(missing_user_ids)} users in user_mapping. Auto-repairing via fit_partial.")

            all_new_user_ids = list(set(new_user_ids + missing_user_ids))

            # Thêm người dùng và tài liệu mới vào dataset
            self.lightfm_dataset.fit_partial(
                users=all_new_user_ids,
                items=self.document_data['document_id'].unique(),
                item_features=self.document_data['category'].explode().unique()
            )
            self.user_mapping, self.reverse_user_mapping, self.item_mapping, self.item_feature_mapping = self.lightfm_dataset.mapping()

            print(f"User mapping {self.user_mapping}")
            print(f"Item mapping {self.item_mapping}")

            # Tạo lại user-item matrix và item features
            user_item_matrix = self.lightfm_dataset.build_interactions(
                [(row['account_id'], row['document_id'], row['rating'])
                 for _, row in self.user_interactions.dropna(subset=['rating']).iterrows()]
            )[0]

            item_features = self.lightfm_dataset.build_item_features(
                [(doc_id, [self.document_data[self.document_data['document_id'] == doc_id]['category'].iloc[0]])
                 for doc_id in self.document_data['document_id']]
            )

            # Huấn luyện mô hình với dữ liệu mới
            self.lightfm_model.fit(
                user_item_matrix,
                item_features=item_features,
                epochs=30,  # Số epoch nhỏ hơn để tối ưu tốc độ cập nhật
                num_threads=4
            )

            self.update_csv_data(
                folder_path=folder_path,
                new_docs_df=new_docs_df,
                new_interactions_df=new_interactions_df,
                update_docs_df=update_docs_df,
                new_user_ids=all_new_user_ids
            )

            print(f"Item embeddings shape: {self.lightfm_model.item_embeddings.shape}")
            print(f"User embeddings shape: {self.lightfm_model.user_embeddings.shape}")
            print(f"Mô hình đã được cập nhật với {len(new_users)} người dùng và {len(new_documents)} tài liệu mới.")

        except Exception as e:
            print("exception: ",e)
            return ApiResponse.error(message=f"Error {e}", code=400)


