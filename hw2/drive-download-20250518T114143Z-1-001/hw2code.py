import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    feature_vector = np.asarray(feature_vector)
    target_vector = np.asarray(target_vector)

    order = np.argsort(feature_vector)
    X_sorted = feature_vector[order]
    y_sorted = target_vector[order]

    diff = np.diff(X_sorted)
    possible_split = diff != 0
    thresholds = (X_sorted[:-1][possible_split] + X_sorted[1:][possible_split]) / 2

    if len(thresholds) == 0:
        return np.array([]), np.array([]), None, None

    y_cumsum = np.cumsum(y_sorted)
    n = len(y_sorted)
    idx = np.where(possible_split)[0]

    left_count = idx + 1
    right_count = n - left_count

    left_ones = y_cumsum[idx]
    left_zeros = left_count - left_ones
    right_ones = y_cumsum[-1] - left_ones
    right_zeros = right_count - right_ones

    with np.errstate(divide='ignore', invalid='ignore'):
        p1_left = left_ones / left_count
        p0_left = left_zeros / left_count
        p1_right = right_ones / right_count
        p0_right = right_zeros / right_count

        H_left = 1 - p1_left**2 - p0_left**2
        H_right = 1 - p1_right**2 - p0_right**2

        Q = -(left_count / n) * H_left - (right_count / n) * H_right

    valid = (left_count > 0) & (right_count > 0)
    thresholds = thresholds[valid]
    Q = Q[valid]

    if len(Q) == 0:
        return np.array([]), np.array([]), None, None

    min_idx = np.argmin(Q)
    threshold_best = thresholds[min_idx]
    gini_best = Q[min_idx]

    return thresholds, Q, threshold_best, gini_best


class DecisionTree:
  def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    if np.any(list(map(lambda x: x != 'real' and x != 'categorical', feature_types))):
      raise ValueError('There is unknown feature type')

    self._tree = {}
    self._depth = 0
    self._feature_types = feature_types
    self._max_depth = max_depth
    self._min_samples_split = min_samples_split
    self._min_samples_leaf = min_samples_leaf

  def _fit_node(self, sub_X, sub_y, node, depth = 0):
    if np.all(sub_y == sub_y[0]): ## fixed uneq
      node['type'] = 'terminal'
      node['class'] = sub_y[0]
      return
    
    if (self._max_depth is not None and depth >= self._max_depth) or \
                len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

    feature_best, threshold_best, gini_best, split = None, None, None, None
    for feature in range(1, sub_X.shape[1]):
      feature_type = self._feature_types[feature]
      categories_map = {}

      if feature_type == 'real':
        feature_vector = sub_X[:, feature]
      elif feature_type == 'categorical':
        counts = Counter(sub_X[:, feature])
        clicks = Counter(sub_X[sub_y == 1, feature])
        ratio = {}
        for key, current_count in counts.items():
          if key in clicks:
            current_click = clicks[key]
          else:
            current_click = 0
          ratio[key] = current_click / current_count # fixed operand order
        
        sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))) # fixed idx
        categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
        feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
      else:
        raise ValueError

      _, _, threshold, gini = find_best_split(feature_vector, sub_y)
      if gini is None:
        continue

      if gini_best is None or gini > gini_best:
        feature_best = feature
        gini_best = gini
        left_indices = feature_vector < threshold
        right_indices = feature_vector > threshold
        split = feature_vector < threshold

        if feature_type == 'real':
          threshold_best = threshold
        elif feature_type == 'categorical':
          threshold_best = list(map(lambda x: x[0],
                        filter(lambda x: x[1] < threshold, categories_map.items())))
        else:
          raise ValueError

    if feature_best is None:
      node['type'] = 'terminal'
      node['class'] = Counter(sub_y).most_common(1)[0][0] # fixed 
      return

    left_mask = split
    right_mask = ~split

    if np.sum(left_mask) < self._min_samples_leaf or np.sum(right_mask) < self._min_samples_leaf:
        node["type"] = "terminal"
        node["class"] = Counter(sub_y).most_common(1)[0][0]
        return

    node['type'] = 'nonterminal'

    node['feature_split'] = feature_best
    if self._feature_types[feature_best] == 'real':
      node['threshold'] = threshold_best
    elif self._feature_types[feature_best] == 'categorical':
      node['categories_split'] = threshold_best
    else:
      raise ValueError
    node['left_child'], node['right_child'] = {}, {}
   
    self._depth = max(self._depth, depth+1)
    self._fit_node(sub_X[left_indices], sub_y[left_indices], node['left_child'], depth + 1)
    self._fit_node(sub_X[right_indices], sub_y[right_indices], node['right_child'], depth + 1) # here only right elems

  def _predict_node(self, x, node):
    if node['type'] == 'terminal':
      return node['class']

    feature_split = node['feature_split']

    if self._feature_types[feature_split] == 'real':
      go_left = x[feature_split] < node['threshold']
    elif self._feature_types[feature_split] == 'categorical':
      go_left = x[feature_split] in node['categories_split']
    else:
      raise ValueError

    if go_left:
      return self._predict_node(x, node['left_child'])
    else:
      return self._predict_node(x, node['right_child'])

  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)
    
    self._tree = {}
    self._fit_node(X, y, self._tree)

  def predict(self, X):
    X = np.array(X)
    predicted = []
    for x in X:
      predicted.append(self._predict_node(x, self._tree))
    return np.array(predicted)
