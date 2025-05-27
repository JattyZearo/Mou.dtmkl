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
from dataloader import *


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
# ============== 3. 数据加载函数 ==============
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils import check_random_state
import os
from sklearn.utils import shuffle


# ============== 4. DTMKL ==============
from DTMKL import *
def compute_A_T_AT(data,label,X_target_unlabeled,y_test,C,A_T_acc,name):
    print(f"\n===== Testing {name}_C={C} =====")
    try_K_li = linear_kernel(data)

    try_svm = SVC(kernel='precomputed', C=C)
    try_svm.fit(try_K_li, label)

    try_kernel_test = np.dot(X_target_unlabeled, data.T)
    try_result = try_svm.predict(try_kernel_test)
    best = accuracy_score(y_test,try_result)
    print(f"try:{accuracy_score(y_test, try_result)}")
    for de in [1.5, 2.0]:
        try_K_li = polynomial_kernel(data, degree=de)

        try_svm = SVC(kernel='precomputed', C=C)
        try_svm.fit(try_K_li, label)

        try_kernel_test = np.dot(X_target_unlabeled, data.T)
        try_result = try_svm.predict(try_kernel_test)
        if accuracy_score(y_test, try_result) > best:
            best = accuracy_score(y_test, try_result)
        print(f"try:{accuracy_score(y_test, try_result)}")
    A_T_acc[name].append(best)
# ============== extra. 对比模块 ==============
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from cvxopt import matrix, solvers
from extra_compare_visualize import *
# ============== 5. 完整实验流程 ==============
if __name__ == "__main__":
    np.random.seed(42)
    # 配置实验参数
    config = {
        "auxiliary": {
            "positive": ["comp.windows.x"],
            "negative": ["rec.sport.hockey"]
        },
        "target": {
            "positive": ["comp.sys.ibm.pc.hardware"],
            "negative": ["rec.motorcycles"]
        }
    }
    m_test = [1,3,5,7,10]  # 每个类别的标记样本数
    n_runs = 3  # 实验重复次数
    svd_dim = 500
    C_values = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    #C_values = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]  # 需要测试的C值
    results = {C: [] for C in C_values}
    models_list = ['DTMKL_f', 'FR', 'A-SVM', 'CD-SVM', 'KMM']
    multi_results = {C: {} for C in C_values}
    A_T_results = {C: {} for C in C_values}
    A_T_list = ['A','T','AT']
    compare = 1 #要不要对比试验
    SVM_A = 0 #要不要基础SVM对比
    for m in m_test:
        for C in C_values:
            print(f"\n===== Testing C={C} =====")
            # 运行实验
            accuracies = []
            multi_acc = {name: [] for name in models_list}
            A_T_acc = {name: [] for name in A_T_list}
            for seed in range(n_runs):
                print(f"\n===== Run {seed + 1}/{n_runs} =====")

                # 1. 加载原始文本数据
                X_aux_text, y_aux, X_target_text, y_target_labeled, X_test_text, y_test = \
                    load_data_ori(config, m=m, random_state=seed)

                # 2. 特征处理（确保一致性）
                processor = FeatureProcessor(svd_dim=svd_dim)
                all_text = X_aux_text + X_target_text + X_test_text
                processor.fit_transform(all_text)

                # 转换各部分数据
                X_aux_feat = processor.transform(X_aux_text)
                X_target_feat = processor.transform(X_target_text)
                X_test_feat = processor.transform(X_test_text)

                # model = SentenceTransformer('all-mpnet-base-v2')
                # X_aux_feat = model.encode(X_aux_text, show_progress_bar=True)
                # X_target_feat = model.encode(X_target_text, show_progress_bar=True)
                # X_test_feat = model.encode(X_test_text, show_progress_bar=True)
                # feat_extractor = BetterFeatureExtractor
                # if feat_extractor:
                #     X_aux_feat = feat_extractor.extract_feature_for_multiple_exs(X_aux_text)
                #     X_target_feat = feat_extractor.extract_feature_for_multiple_exs(X_target_text)
                #     X_test_feat = feat_extractor.extract_feature_for_multiple_exs(X_test_text)
                # else:
                #     train_feat = None

                # 3. 划分目标域标记/未标记数据
                n_labeled = 2 * m
                X_target_labeled = X_target_feat[:n_labeled]
                X_target_unlabeled = X_target_feat[n_labeled:]

                #try 模型对比

                if SVM_A == 1:
                    compute_A_T_AT(X_aux_feat,y_aux,X_target_unlabeled,y_test,C,A_T_acc,'A')
                    compute_A_T_AT(X_target_labeled, y_target_labeled, X_target_unlabeled, y_test, C,A_T_acc,'T')
                    compute_A_T_AT(np.concatenate([X_aux_feat, X_target_labeled], axis=0),np.concatenate([y_aux, y_target_labeled], axis=0),X_target_unlabeled, y_test,C,A_T_acc,'AT')
                if compare == 1:
                    # 训练和评估 FR
                    fr_model = FeatureReplicationSVM(C=C)
                    fr_model.fit(X_aux_feat, y_aux, X_target_labeled, y_target_labeled)
                    y_pred_fr = fr_model.predict(X_test_feat)
                    multi_acc['FR'].append(accuracy_score(y_test, y_pred_fr))
                    print("FR Accuracy:", accuracy_score(y_test, y_pred_fr))

                    # 训练和评估 A-SVM
                    a_svm = AdaptiveSVM(C=C)
                    a_svm.fit(X_aux_feat, y_aux, X_target_labeled, y_target_labeled)
                    y_pred_asvm = a_svm.predict(X_test_feat)
                    multi_acc['A-SVM'].append(accuracy_score(y_test, y_pred_asvm))
                    print("A-SVM Accuracy:", accuracy_score(y_test, y_pred_asvm))

                    # 训练和评估 CD-SVM
                    cd_svm = CDSVM(C=C, k=5)
                    cd_svm.fit(X_aux_feat, y_aux, X_target_labeled, y_target_labeled)
                    y_pred_cdsvm = cd_svm.predict(X_test_feat)
                    multi_acc['CD-SVM'].append(accuracy_score(y_test, y_pred_cdsvm))
                    print("CD-SVM Accuracy:", accuracy_score(y_test, y_pred_cdsvm))

                    # 训练和评估 KMM（需结合SVM）
                    kmm = KMM(gamma=0.1, B=0.9)
                    kmm.fit(X_aux_feat, X_target_labeled, X_target_unlabeled)
                    weights = kmm.get_weights()

                    # 使用KMM权重训练SVM
                    svm = SVC(C=C, kernel='linear')
                    X_train = np.vstack([X_aux_feat, X_target_labeled])
                    y_train = np.concatenate([y_aux, y_target_labeled])
                    sample_weight = np.concatenate([weights, np.ones(len(y_target_labeled))])
                    svm.fit(X_train, y_train, sample_weight=sample_weight)
                    y_pred_kmm = svm.predict(X_test_feat)
                    multi_acc['KMM'].append(accuracy_score(y_test, y_pred_kmm))
                    print("KMM Accuracy:", accuracy_score(y_test, y_pred_kmm))

                # 4. 训练模型
                model = DTMKL_f(C=C)
                model.fit(
                    X_aux_feat,
                    y_aux,
                    X_target_labeled,
                    y_target_labeled,
                    X_target_unlabeled
                )

                n_labeled = len(y_target_labeled)
                n_aux = len(y_aux)
                # 5. 预测评估
                y_pred_continuous = model.predict(n_labeled, n_aux)
                y_pred = np.where(y_pred_continuous >= 0, 1, -1)
                acc = accuracy_score(y_test, y_pred)
                accuracies.append(acc)
                multi_acc['DTMKL_f'].append(acc)
                print(f"Run {seed + 1} Accuracy: {acc:.4f}")
            results[C] = accuracies
            if compare == 1:
                multi_results[C] = multi_acc
            if SVM_A == 1:
                A_T_results[C] = A_T_acc

        plot_result = {name: ([], []) for name in models_list}
        print("\n===== Final Results =====")
        for C in C_values:

            mean_acc = np.mean(results[C])
            std_acc = np.std(results[C])
            print(f"C={C}: Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")
            if compare == 1:
                for r in models_list:
                    mean_acc_m = np.mean(multi_results[C][r])
                    std_acc_m = np.std(multi_results[C][r])

                    plot_result[r][0].append(mean_acc_m)
                    plot_result[r][1].append(std_acc_m)
                    print(f"C={C}: model-{r}Accuracy = {mean_acc_m:.4f} ± {std_acc_m:.4f}")
            if SVM_A == 1:
                for r in A_T_list:
                    mean_acc_at = np.mean(A_T_results[C][r])
                    std_acc_at = np.std(A_T_results[C][r])
                    print(f"C={C}: model-{r}Accuracy = {mean_acc_at:.4f} ± {std_acc_at:.4f}")
        if compare == 1:
            draw_pic(models_list, plot_result, C_values,m)
