# ============== extra. 对比模块 ==============
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
class FeatureReplicationSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.clf = SVC(C=C, kernel='linear')

    def _expand_features(self, X, domain='aux'):
        """特征扩展：辅助域[原特征, 原特征, 0], 目标域[原特征, 0, 原特征]"""
        n_samples, n_features = X.shape
        if domain == 'aux':
            return np.hstack([X, X, np.zeros((n_samples, n_features))])
        elif domain == 'target':
            return np.hstack([X, np.zeros((n_samples, n_features)), X])

    def fit(self, X_aux, y_aux, X_target_labeled, y_target_labeled):
        # 合并数据并扩展特征
        X_aux_expanded = self._expand_features(X_aux, domain='aux')
        X_target_expanded = self._expand_features(X_target_labeled, domain='target')
        X_combined = np.vstack([X_aux_expanded, X_target_expanded])
        y_combined = np.concatenate([y_aux, y_target_labeled])

        self.clf.fit(X_combined, y_combined)

    def predict(self, X_test):
        X_test_expanded = self._expand_features(X_test, domain='target')
        return self.clf.predict(X_test_expanded)


from sklearn.svm import SVC
class AdaptiveSVM:
    def __init__(self, C=1.0):
        self.base_clf = SVC(C=C, kernel='linear', probability=True)  # 启用概率输出
        self.delta_clf = SVR(C=C, kernel='linear')  # 使用回归模型拟合连续残差

    def fit(self, X_aux, y_aux, X_target_labeled, y_target_labeled):
        # 训练基分类器
        self.base_clf.fit(X_aux, y_aux)

        # 获取基分类器在目标域标记数据上的概率预测（映射到[-1, 1]）
        f_base_prob = self.base_clf.predict_proba(X_target_labeled)[:, 1] * 2 - 1

        # 计算残差（连续值）
        residual = y_target_labeled - f_base_prob

        # 训练调整模型
        self.delta_clf.fit(X_target_labeled, residual)

    def predict(self, X_test):
        # 基分类器的概率预测
        base_prob = self.base_clf.predict_proba(X_test)[:, 1] * 2 - 1

        # 调整项的预测
        delta = self.delta_clf.predict(X_test)

        # 综合预测
        return np.sign(base_prob + delta)

from sklearn.neighbors import NearestNeighbors


class CDSVM:
    def __init__(self, C=1.0, k=5):
        self.C = C
        self.k = k
        self.clf = SVC(C=C, kernel='linear', class_weight='balanced')

    def fit(self, X_aux, y_aux, X_target_labeled, y_target_labeled):
        # 合并所有目标域标记数据作为参考
        X_target_all = np.vstack([X_target_labeled])

        # 计算辅助域样本的权重：与目标域的相似度
        if len(y_target_labeled) == 2:
            knn = NearestNeighbors(n_neighbors=1).fit(X_target_all)
        else:
            knn = NearestNeighbors(n_neighbors=self.k).fit(X_target_all)
        distances, _ = knn.kneighbors(X_aux)
        weights = 1.0 / (np.mean(distances, axis=1) + 1e-6)
        weights /= np.max(weights)  # 归一化

        # 合并数据并加权训练
        X_train = np.vstack([X_aux, X_target_labeled])
        y_train = np.concatenate([y_aux, y_target_labeled])
        sample_weight = np.concatenate([weights, np.ones(len(y_target_labeled))])

        self.clf.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X_test):
        return self.clf.predict(X_test)


from cvxopt import matrix, solvers


class KMM:
    def __init__(self, kernel='rbf', gamma=0.1, B=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.B = B  # 权重边界

    def fit(self, X_aux, X_target_labeled, X_target_unlabeled):
        # 合并目标域数据（标记+未标记）
        X_target = np.vstack([X_target_labeled, X_target_unlabeled])

        # 计算核矩阵
        K = rbf_kernel(X_aux, X_target, gamma=self.gamma)
        n_aux = X_aux.shape[0]
        n_target = X_target.shape[0]

        # 优化目标：最小化 ||Phi(aux)*beta - Phi(target)||
        # 转换为二次规划问题
        K_aux = rbf_kernel(X_aux, gamma=self.gamma)
        K_target = rbf_kernel(X_target, gamma=self.gamma)

        P = matrix(K_aux)
        q = -matrix(np.mean(K, axis=1))
        G = matrix(np.vstack([-np.eye(n_aux), np.eye(n_aux)]))  # 0 <= beta_i <= B
        h = matrix(np.hstack([np.zeros(n_aux), np.ones(n_aux) * self.B]))
        A = matrix(np.ones((1, n_aux)), (1, n_aux))
        b = matrix(n_target * 1.0, (1, 1))

        sol = solvers.qp(P, q, G, h, A, b)
        self.beta = np.array(sol['x']).flatten()

    def get_weights(self):
        return self.beta


# ================== extra. 绘图 ==================
def draw_pic(models, results, C_values,m):
    # ================== 1. 数据 ==================
    if m != 5 and m != 10:
        return
    # ================== 2. 绘图配置 ==================
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 12})

    # 颜色和标记样式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    # ================== 3. 绘制折线图 ==================
    for idx, model in enumerate(models):
        means, stds = results[model]

        # 绘制带误差线的折线
        plt.errorbar(
            x=C_values,
            y=means,
            yerr=stds,
            color=colors[idx],
            marker=markers[idx],
            markersize=8,
            linewidth=2,
            capsize=5,
            capthick=2,
            elinewidth=2,
            label=model
        )

    # ================== 4. 图形美化 ==================
    #plt.xscale('log')  # C值通常用对数刻度
    plt.xticks(C_values, labels=[str(c) for c in C_values])
    plt.xlabel('Regularization Parameter C', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Classification Performance Comparison with Varying C', fontsize=16)
    plt.legend(loc='lower right', frameon=True, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.5, 1)  # 根据实际数据调整范围

    # ================== 5. 保存/显示图像 ==================
    plt.tight_layout()
    plt.savefig('accuracy_vs_C_rec_'+str(m)+'.png', dpi=300)
    plt.show()
