# ============== 1. 依赖库导入 ==============
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.preprocessing import Normalizer

solvers.options['show_progress'] = False

# ============== 2. 特征处理器类 ==============
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion
from sentence_transformers import SentenceTransformer
from feature_extractor import BoWFeatureExtractor, FeatureExtractor, BetterFeatureExtractor
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')


def clean_text(text):
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除特殊字符和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 转换为小写
    text = text.lower()
    # 分词
    words = nltk.word_tokenize(text)
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # 词干提取
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # 合并并去除多余空格
    return ' '.join(words).strip()


class FeatureProcessor:
    """确保训练/测试使用相同的特征转换器"""

    def __init__(self, max_features=15000, svd_dim=5000):
        # self.tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        # self.svd = TruncatedSVD(n_components=svd_dim)
        # self.pipeline = Pipeline([
        #     ('tfidf', TfidfVectorizer(max_features=max_features, stop_words='english')),
        #     ('svd', TruncatedSVD(svd_dim))
        # ])
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, stop_words='english')),
            ('normalizer', Normalizer(norm='l2'))
        ])

    def fit_transform(self, texts):
        return self.pipeline.fit_transform(texts).toarray()

    def transform(self, texts):
        return self.pipeline.transform(texts).toarray()


# ============== 3. 数据加载函数 ==============
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils import check_random_state
import os
from sklearn.utils import shuffle


def load_data_ori(config, m=5, random_state=42):
    """返回原始文本数据（未转换）"""
    # 加载辅助域数据
    aux_pos = fetch_20newsgroups(
        subset='train',
        categories=config["auxiliary"]["positive"],
        shuffle=True,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )
    aux_pos2 = fetch_20newsgroups(
        subset='test',
        categories=config["auxiliary"]["positive"],
        shuffle=True,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )

    aux_neg = fetch_20newsgroups(
        subset='train',
        categories=config["auxiliary"]["negative"],
        shuffle=True,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )
    aux_neg2 = fetch_20newsgroups(
        subset='test',
        categories=config["auxiliary"]["negative"],
        shuffle=True,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )
    aux_pos.data = [clean_text(text) for text in aux_pos.data]
    aux_neg.data = [clean_text(text) for text in aux_neg.data]

    X_aux_text = aux_pos.data + aux_pos2.data + aux_neg.data + aux_neg2.data
    y_aux = np.array([1] * (len(aux_pos.data) + len(aux_pos2.data)) + [-1] * (len(aux_neg.data) + len(aux_neg2.data)))

    # 加载目标域全量数据
    target_pos = fetch_20newsgroups(
        subset='train',
        categories=config["target"]["positive"],
        shuffle=True,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )
    target_neg = fetch_20newsgroups(
        subset='train',
        categories=config["target"]["negative"],
        shuffle=True,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )
    target_pos.data = [clean_text(text) for text in target_pos.data]
    target_neg.data = [clean_text(text) for text in target_neg.data]
    # 随机抽取m个正/负样本作为标记数据
    np.random.seed(random_state)
    pos_idx = np.random.choice(len(target_pos.data), m, replace=False)
    neg_idx = np.random.choice(len(target_neg.data), m, replace=False)

    # 目标域标记文本和标签
    X_target_labeled_text = [target_pos.data[i] for i in pos_idx] + \
                            [target_neg.data[i] for i in neg_idx]
    y_target_labeled = np.array([1] * m + [-1] * m)

    # 目标域未标记文本（剩余样本）
    X_target_unlabeled_text = [
                                  target_pos.data[i] for i in range(len(target_pos.data)) if i not in pos_idx
                              ] + [
                                  target_neg.data[i] for i in range(len(target_neg.data)) if i not in neg_idx
                              ]

    # 测试集数据
    X_test_text = X_target_unlabeled_text
    y_test = np.array([1] * (len(target_pos.data) - m) + [-1] * (len(target_neg.data) - m))

    # 合并目标域所有文本
    X_target_text = X_target_labeled_text + X_target_unlabeled_text

    return X_aux_text, y_aux, X_target_text, y_target_labeled, X_test_text, y_test


def load_data(config, m=5, random_state=42):
    """从本地20news-bydate文件夹加载数据"""
    # 定义数据路径
    base_path = "20news-bydate"
    train_path = os.path.join(base_path, "20news-bydate-train")
    test_path = os.path.join(base_path, "20news-bydate-test")

    # 加载辅助域数据（从训练集）
    def load_category(categories, subset='train'):
        data = []
        for category in categories:
            dir_path = os.path.join(train_path if subset == 'train' else test_path, category)
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.isdigit()]
            for file_path in files:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                    text = clean_text(text)
                    data.append(text)
        return data

    # 辅助域正负样本
    aux_pos = load_category(config["auxiliary"]["positive"], subset='train')
    aux_neg = load_category(config["auxiliary"]["negative"], subset='train')
    X_aux_text = aux_pos + aux_neg
    y_aux = np.array([1] * len(aux_pos) + [-1] * len(aux_neg))

    # 目标域全量数据（从训练集）
    target_pos = load_category(config["target"]["positive"], subset='train')
    target_neg = load_category(config["target"]["negative"], subset='train')

    # 随机抽取m个正/负样本作为标记数据
    np.random.seed(random_state)
    pos_idx = np.random.choice(len(target_pos), m, replace=False)
    neg_idx = np.random.choice(len(target_neg), m, replace=False)

    # 目标域标记文本和标签
    X_target_labeled_text = [target_pos[i] for i in pos_idx] + [target_neg[i] for i in neg_idx]
    y_target_labeled = np.array([1] * m + [-1] * m)

    # 目标域未标记文本（剩余样本）
    X_target_unlabeled_text = [
                                  target_pos[i] for i in range(len(target_pos)) if i not in pos_idx
                              ] + [
                                  target_neg[i] for i in range(len(target_neg)) if i not in neg_idx
                              ]

    # 测试集数据
    X_test_text = X_target_unlabeled_text
    y_test = np.array([1] * (len(target_pos) - m) + [-1] * (len(target_neg) - m))

    # 合并目标域所有文本
    X_target_text = X_target_labeled_text + X_target_unlabeled_text

    return X_aux_text, y_aux, X_target_text, y_target_labeled, X_test_text, y_test
