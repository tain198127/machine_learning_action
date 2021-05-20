import numpy as np

raw_data = []
user_ids = set()
item_ids = set()
user_id_to_idx = {}
user_idx_to_id = {}
item_id_to_idx = {}
item_idx_to_id = {}
# 读取数据
with open(file="data.csv", mode="r", encoding="utf8") as f:
    f.readline()
    while True:
        line = f.readline()
        if line:
            # 读一行
            temp = line.strip().split(",")
            # 原始数据
            raw_data.append(temp)
            # 用户id
            user_ids.add(temp[0])
            # 商品id
            item_ids.add(temp[1])
        else:
            break

user_num = len(user_ids)
item_num = len(item_ids)
# 转化成矩阵的行和列，将id转化为int
for idx, user_id in enumerate(user_ids):
    user_id_to_idx[user_id] = idx
    user_idx_to_id[idx] = user_id

for idx, item_id in enumerate(item_ids):
    item_id_to_idx[item_id] = idx
    item_idx_to_id[idx] = item_id
# 用户-商品-评分矩阵
user_item_rating_matrix = np.zeros(shape=(user_num, item_num), dtype="int")

# 每个用户购买的商品列表
user_item_list = []
for idx in range(user_num):
    user_item_list.append(set())
# 构建矩阵
for row in raw_data:
    # 求行坐标，用户索引
    user_idx = user_id_to_idx[row[0]]
    # 求列坐标，商品索引
    item_idx = item_id_to_idx[row[1]]
    # 求分数
    score = int(row[2])
    # 赋值
    user_item_rating_matrix[user_idx, item_idx] = score
    # 购买过的产品
    user_item_list[user_idx].add(item_idx)

# 被推荐的用户
input_user_id = '3a325c58b5d911ebacb7309c231931a5'
input_user_idx = user_id_to_idx[input_user_id]
input_user_item = user_item_list[input_user_idx]
input_user_times_id = [item_idx_to_id[idx] for idx in input_user_item]
# print(input_user_times_id)
# 用户相似度数组
user_similarity_arr = np.zeros(shape=user_num, )
# 遍历所有用户，求待推荐的用户id
for idx in range(user_num):
    if idx == input_user_idx:
        continue
    else:
        cur_user_items = user_item_list[idx]
        # 公共产品交集
        inter_items = cur_user_items.intersection(input_user_item)
        # 公共交集部分，超过10个，就认为是亲密的。这个地方是超参
        if len(inter_items) < 10:
            continue
        # 矩阵分解 SVD是一种方式；另外一种思想是KNN；
        # 求出当前用户产品评分向量,idx表示的是用户索引，inter_items表示公共交集部分，这个是numpy的能力
        cur_user_vector = user_item_rating_matrix[idx, list(inter_items)]
        # 待推荐用户的产品评分向量
        input_user_vector = user_item_rating_matrix[input_user_idx, list(inter_items)]

        # 求两个用户-产品-向量的余弦相似度
        cosin_similarity = np.dot(cur_user_vector, input_user_vector) / np.linalg.norm(
            cur_user_vector) / np.linalg.norm(input_user_vector)
        # 放入
        #         user_similarity_arr[idx] = np.round(cosin_similarity,3)
        user_similarity_arr[idx] = cosin_similarity
        # 业务能力转化为程序问题，理解数学的内涵，把公式写成算法，可以先死记硬背

# 对数据进行排序，然后返回的跟他最相似的50个人
most_similarity_user_idx = user_similarity_arr.argsort()[::-1][:50]

# 相似50人的相似度
most_sim_score = user_similarity_arr[most_similarity_user_idx]
# 这里还是正确的
most_sim_user_id = [user_idx_to_id[idx] for idx in most_similarity_user_idx]
# print(most_sim_user_id)
print('--------------------------最相似50人评分')
print(most_sim_score)
print('--------------------------最相似50人的ID')
print(most_sim_user_id)
# 推荐的20个商品

rec_items = np.zeros(shape=(item_num,))
# 遍历所有产品，进行加权平均分，倒叙的前20个
for idx in range(item_num):
    if idx in input_user_item:
        continue
    # 拿到所有与这个产品相关的人——相关人
    all_related_users = np.where(user_item_rating_matrix[:, idx] > 0)[0]
    # 相关人员和最近相似人做交集，超过2个的，才能推荐
    inter_users = np.intersect1d(most_similarity_user_idx, all_related_users)
    if inter_users.size < 2:
        continue
    # 拿到购买过这个产品，并且是相关人员的相似度
    inter_user_sim = user_similarity_arr[inter_users]
    # 分数
    inter_user_rating = user_item_rating_matrix[:, idx][inter_users]
    # 加权平均评分
    rec_items[idx] = np.dot(inter_user_sim, inter_user_rating) / np.sum(inter_user_sim)
# 待推荐的产品的索引
rec_idxes = rec_items.argsort()[::-1][:20]
print('--------------------------推荐的20个的评分')
print(rec_items[rec_idxes])
rec_ids = [item_idx_to_id[idx] for idx in rec_idxes]
print('--------------------------推荐的20个产品ID')
print(rec_ids)

