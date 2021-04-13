import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from time import time
from math import inf


def calc_impurity(target_vector):
    p1 = sum(target_vector) / len(target_vector)
    p0 = 1 - p1
    return 1 - (p1 ** 2) - (p0 ** 2)


def calc_quality(feature_vector, target_vector, threshold, min_samples = None):
    
    feature_target = np.array([feature_vector, target_vector])
    
    cond = feature_target[0] < threshold
    
    left = feature_target[:, feature_target[0] < threshold]
    right = feature_target[:, feature_target[0] >= threshold]
    
    if ((min_samples != None) and ((left.shape[1] < min_samples) or (right.shape[1] < min_samples))):
        return -inf
    
    ratio_l = left.shape[1] / feature_target.shape[1]
    ratio_r = right.shape[1] / feature_target.shape[1]
    
    
    return -(ratio_l * calc_impurity(left[1])) - (ratio_r * calc_impurity(right[1]))
    

def find_best_split(feature_vector, target_vector, min_samples_leaf = None):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
    $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector = np.asarray(feature_vector)
    target_vector = np.asarray(target_vector)
    
    feature_target = np.array([feature_vector, target_vector])
    
    sorted_feature_vector = np.array(sorted(set(feature_vector)))
    
    thresholds = 0.5*(sorted_feature_vector[1:] + sorted_feature_vector[:-1])
    
    #Не знаю, как без этого обойтись. В какие только дебри документации numpy не влезал, ничего не нашел.
    #Загадка Жанны Фриске, которую не могу решить.
    ginis = np.asarray([calc_quality(feature_vector, target_vector, threshold, min_samples_leaf) for threshold in thresholds])
    
    threshold_best = thresholds[np.argmax(ginis)]
    gini_best = np.max(ginis)
    
    return (thresholds, ginis, threshold_best, gini_best)


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self.tree = {}
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth = 0):
        
        sub_X = np.asarray(sub_X)
        sub_y = np.asarray(sub_y)
        
        if ((np.all(sub_y == sub_y[0])) or ((self.max_depth != None) and (depth > self.max_depth)) or ((self.min_samples_split != None) and (len(sub_y) < self.min_samples_split))):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(1, sub_X.shape[1]):
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))

            else:
                raise ValueError
                
            feature_vector = np.asarray(feature_vector)
            
            if np.all(feature_vector == feature_vector[0]):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y, self.min_samples_leaf)
            if ((gini_best is None or gini > gini_best) and (gini != -inf)):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = set(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        x = np.asarray(x)
        feature = node["feature_split"]
        
        if self.feature_types[feature] == "real":
            left_split = x[feature] < node["threshold"]
        elif self.feature_types[feature] == "categorical":
            left_split = x[feature] in node["categories_split"]
        else:
            raise ValueError
        
        if left_split:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])
            
        

    def fit(self, X, y):
        self._fit_node(X, y, self.tree)

    def predict(self, X):
        predicted = []
        X = np.asarray(X)
        for x in X:
            predicted.append(self._predict_node(x, self.tree))
        return np.array(predicted)