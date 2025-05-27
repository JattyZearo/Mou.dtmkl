# ============== 4. DTMKL_f模型类 ==============
class DTMKL_f():
    """使用现有基分类器的DTMKL_f"""

    def __init__(self, C=1.0, lr=0.05, zeta=0.1, theta=1e-5, max_iter=6, lambda_=0.5, tol=1e-3):
        self.base_kernels = []  # 基核列表
        self.C = C  # SVM正则化参数
        self.zeta = zeta  # 目标域标记数据正确和未标记数据上分类器相似度正则项权重
        self.theta = theta  # J(d)正则项权重
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.d = None  # 核组合系数
        self.svr = None  # SVR
        self.lr = lr  # SVR学习率
        self.base_classifiers = None
        self.lambda_ = lambda_  # 未标记数据上分类器相似度正则项权重

    def generate_base_kernels(self, X_aux, X_target):
        """生成多个基核矩阵（线性核、多项式核等）"""
        base_kernels = []
        # 线性核
        K_linear = linear_kernel(np.concatenate([X_aux, X_target], axis=0))
        base_kernels.append(K_linear)
        # 多项式核（不同次数）
        for degree in [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
            K_poly = polynomial_kernel(np.concatenate([X_aux, X_target], axis=0), degree=degree)
            base_kernels.append(K_poly)
        return base_kernels

    def base_classifier_fit(self, y):
        base_classifiers = []
        for index, K_m in enumerate(self.base_kernels):
            svm = SVR(kernel='precomputed', C=self.C)
            svm.fit(K_m[:len(y), :len(y)], y)
            base_classifiers.append(svm)
        return base_classifiers

    def _compute_mmd(self, K, s):
        """K: (n_total, n_total), s: (n_total,)"""
        return np.trace(K @ np.outer(s, s))

    def fit(self, X_aux, y_aux, X_labeled, y_labeled, X_unlabeled):
        X_target = np.concatenate([X_labeled, X_unlabeled], axis=0)
        X_train = np.concatenate([X_aux, X_labeled], axis=0)
        y_train = np.concatenate([y_aux, y_labeled], axis=0)
        self.base_kernels = self.generate_base_kernels(X_aux, X_target)

        self.base_classifiers = self.base_classifier_fit(y_train)

        X_target_combined = np.vstack([X_labeled, X_unlabeled])
        n_A = X_aux.shape[0]
        n_Tl = X_labeled.shape[0]
        n_Tu = X_unlabeled.shape[0]
        self.n_A = n_A
        self.n_Tl = n_Tl
        self.n_Tu = n_Tu
        self.n_T = self.n_Tu + self.n_Tl
        n_T = n_Tl + n_Tu
        n_labeled = X_labeled.shape[0]
        n_unlabeled = X_unlabeled.shape[0]
        n_train = X_aux.shape[0] + X_labeled.shape[0]
        M = len(self.base_classifiers)
        self.d = np.ones(M) / M

        K_combined = sum(self.d[m] * self.base_kernels[m] for m in range(M))

        # 计算MMD向量 p = [tr(K_m S)]
        s = np.concatenate([np.ones(n_A) / n_A, -np.ones(n_T) / n_T])
        S = np.outer(s, s)

        self.base_classifiers = self.base_classifier_fit(y_train)

        # 迭代优化
        for iter in range(self.max_iter):
            # 预计算基分类器在未标记数据上的决策值
            f_base = np.zeros((n_unlabeled, M))

            #kernel_test = np.dot(X_unlabeled, X_train.T)
            for m in range(M):
                K_test = self.base_kernels[m][n_train:, :n_train]  # 未标记数据与训练数据的核矩阵块
                f_base[:, m] = self.base_classifiers[m].predict(K_test)
                #f_base[:, m] = self.base_classifiers[m].predict(kernel_test)

            y_virtual = f_base @ self.d

            mmd_values = [np.trace(K_m @ S) for K_m in self.base_kernels]

            p = np.array(mmd_values)
            #K_combined = np.einsum('m,mij->ij', self.d, self.base_kernels)
            #mmd = self._compute_mmd(K_combined, s)

            # 步骤1：训练SVR（使用标记数据）
            combined_kernel = sum(self.d[m] * self.base_kernels[m] for m in range(M))
            self.svr = SVR(kernel='precomputed', C=self.C, epsilon=0.1)
            self.svr.fit(combined_kernel, np.concatenate([y_train, y_virtual], axis=0))

            # 步骤2：计算梯度并更新d
            alpha_diff = self.svr.dual_coef_  # 形状 (1, n_SV)
            alpha_sum = np.abs(alpha_diff).sum()

            grad_J = self._compute_grad_J_new(alpha_diff, alpha_sum, f_base)
            grad_mmd = p.T @ p * self.d  # 同DTMKL_AT的MMD梯度计算
            grad_total = grad_mmd + self.theta * grad_J

            self.d -= self.lr * grad_total

            self.d = self._project_to_simplex(self.d)
            #print(f"Iteration {iter + 1}: d = {self.d}")

        combined_kernel = sum(self.d[m] * self.base_kernels[m] for m in range(M))
        self.svr.fit(combined_kernel[:n_A + n_Tl, :n_A + n_Tl], y_train)

    def _compute_mmd_grad(self, mmd, s):
        grad_mmd = np.zeros(len(self.base_classifiers))
        for m in range(len(self.base_classifiers)):
            grad_mmd[m] = 1 / 2 * mmd * np.trace(self.base_kernels[m] @ np.outer(s, s))
        return grad_mmd

    def _project_to_simplex(self, d):
        """投影到单纯形约束：d >=0, sum(d)=1"""
        d = np.where(d < 0, 1 / len(d), d)
        d_sorted = np.sort(d)[::-1]
        cum_sum = np.cumsum(d_sorted) - 1
        idx = np.arange(1, len(d) + 1)
        rho = np.where(d_sorted - cum_sum / idx > 0)[0][-1]
        theta = cum_sum[rho] / (rho + 1)
        return np.maximum(d - theta, 0)

    def generate_diagonal_matrix(self, n, u, lambda_value):
        # 矩阵的总大小
        size = n + u
        # 前n行是1，后u行是1/lambda
        diagonal_values = [1] * n + [1 / lambda_value] * u
        # 生成对角矩阵
        matrix = np.diag(diagonal_values)
        return matrix

    def _compute_grad_J_new(self, alpha_diff, alpha_sum, f_base):
        sv_indices = self.svr.support_
        grad = np.zeros(len(self.base_kernels))
        for m in range(len(self.base_kernels)):
            K_m_grad = self.base_kernels[m]
            K_grad_matrix = self.generate_diagonal_matrix(self.n_A + self.n_Tl, self.n_Tu, lambda_value=self.lambda_)
            K_grad = 1 / self.zeta * 2 * self.d[m] * K_grad_matrix
            K_hat_grad = K_m_grad + K_grad
            K_hat_grad_sv = K_hat_grad[sv_indices][:, sv_indices]
            first_term = -0.5 * alpha_diff @ K_hat_grad_sv @ alpha_diff.T

            y_hat = np.concatenate([np.zeros(self.n_A + self.n_Tl), f_base[:, m]], axis=0)
            second_term = - alpha_diff @ y_hat[sv_indices]
            grad[m] = first_term + second_term

        return grad

    def _compute_grad_J_svr(self, alpha, K_combined, y_aux, y_labeled, f_base):
        """计算SVR结构风险项的梯度"""
        grad = np.zeros(len(self.base_kernels))
        n_labeled = len(y_labeled)
        n_aux = len(y_aux)
        sv_indices = self.svr.support_  # 支持向量的索引

        # 标记数据梯度项
        for m in range(len(self.base_kernels)):
            K_m_labeled = self.base_kernels[m][n_aux:n_aux + n_labeled, n_aux:n_aux + n_labeled]
            K_m_sv = K_m_labeled[sv_indices][:, sv_indices]
            grad[m] = - 0.5 * alpha @ K_m_sv @ alpha.T / 2 / self.d[m] / self.d[m]

        # 未标记数据正则项梯度
        y_virtual = f_base @ self.d
        for m in range(len(self.base_kernels)):
            K_m_unlabeled = self.base_kernels[m][n_aux + n_labeled:, n_aux + n_labeled:]
            residual = y_virtual - self.svr.predict(K_combined[n_aux + n_labeled:, n_aux:n_aux + n_labeled])
            grad[m] += self.lambda_ * self.d[m] * np.dot(residual, K_m_unlabeled @ residual)
        return grad

    def predict(self, n_labeled, n_aux):
        """预测测试集"""

        combined_kernel_test = sum(self.d[m] * self.base_kernels[m] for m in range(len(self.base_kernels)))
        return self.svr.predict(combined_kernel_test[n_aux + n_labeled:, :n_aux + n_labeled])
