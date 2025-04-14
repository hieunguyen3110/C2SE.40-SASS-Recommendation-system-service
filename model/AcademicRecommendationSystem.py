import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


class SearchBasedRecommender(object):
    """Collaborative Filtering (CF) recommender system"""

    def __init__(self, Y_data, k, dist_func=cosine_similarity, uuCF=1):
        self.uuCF = uuCF  # 1: User-User CF, 0: Item-Item CF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]  # Nếu là item-item thì hoán đổi user và item
        self.k = k  # Số lượng lân cận gần nhất
        self.dist_func = dist_func  # Hàm đo độ tương đồng
        self.Ybar_data = None

        # Xác định số lượng người dùng và sản phẩm từ dữ liệu
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def add(self, new_data):
        """Thêm dữ liệu đánh giá mới vào ma trận Y_data"""
        self.Y_data = np.concatenate((self.Y_data, new_data), axis=0)
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1  # Cập nhật số lượng người dùng mới
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1  # Cập nhật số lượng item mới

    def normalize_Y(self):
        """Chuẩn hóa ma trận Y để tạo ma trận trung bình đánh giá của mỗi user"""
        users = self.Y_data[:, 0]
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))  # Mảng lưu trung bình đánh giá của mỗi user

        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)  # Các dòng thuộc user n
            item_ids = self.Y_data[ids, 1]  # Danh sách sản phẩm user n đã đánh giá
            interactive = self.Y_data[ids, 2]  # Điểm đánh giá của user n
            # Tính trung bình
            m = np.mean(interactive) if len(interactive) > 0 else 0
            self.mu[n] = m

            # Chuẩn hóa điểm đánh giá (trừ đi giá trị trung bình của user)
            self.Ybar_data[ids, 2] = interactive - self.mu[n]

        # Tạo ma trận thưa (sparse matrix) để tiết kiệm bộ nhớ
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
                                       (self.Ybar_data[:, 1], self.Ybar_data[:, 0])),
                                      shape=(self.n_items, self.n_users)).tocsr()

    def similarity(self):
        """Tính toán ma trận độ tương đồng giữa các user hoặc giữa các items"""
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

    def refresh(self):
        """Cập nhật lại ma trận sau khi thêm đánh giá mới"""
        self.normalize_Y()
        self.similarity()

    def fit(self):
        """Huấn luyện mô hình"""
        self.refresh()

    def __pred(self, u, i, normalized=True):
        """Dự đoán điểm đánh giá của user u cho item i"""
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)  # Các user đã đánh giá item i
        users_rated_i = self.Y_data[ids, 0].astype(np.int32)
        sim = self.S[u, users_rated_i]  # Tính toán độ tương đồng giữa u và các user đã đánh giá item i

        # Chọn k user giống nhất
        a = np.argsort(sim)[-self.k:]
        nearest_s = sim[a]  # Độ tương đồng của k user gần nhất
        r = self.Ybar[i, users_rated_i[a]]  # Điểm đánh giá của k user gần nhất

        if normalized:
            return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8)
        return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def pred(self, u, i, normalized=True):
        """Hàm dự đoán điểm đánh giá (tùy chọn user-user hoặc item-item CF)"""
        return self.__pred(u, i, normalized) if self.uuCF else self.__pred(i, u, normalized)

    def recommend(self, u):
        """Đề xuất danh sách sản phẩm cho user u dựa trên điểm đánh giá dự đoán"""
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()

        recommended_items = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                score = self.__pred(u, i)
                print("recommend doc "+str(i)+" for user "+str(u)+ " with score is: "+ str(score))
                if score > 0:
                    recommended_items.append(i)

        return recommended_items

    def recommend2(self, u):
        """
        Determine all items should be recommended for user u.
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which
        have not been rated by u yet.
        """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommended_items = []

        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0:
                    recommended_items.append(i)

        return recommended_items

    def print_recommendation(self):
        """
        print all items which should be recommended for each user
        """
        print('Recommendation:')
        for u in range(self.n_users):  # Fix xrange -> range
            recommended_items = self.recommend(u)
            if self.uuCF:
                print(f'    Recommend doc(s): {recommended_items} for user {u}')
            else:
                print(f'    Recommend User {u} for doc(s): {recommended_items}')
