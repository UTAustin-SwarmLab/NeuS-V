import numpy as np


class DataTransformer:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.sorted_data = np.sort(self.data)
        self.n = len(self.sorted_data)
        self.ecdf = np.arange(1, self.n + 1) / self.n

    def mapping_function(self, x):
        x = np.asarray(x)
        return np.interp(x, self.sorted_data, self.ecdf, left=0, right=1)


def smooth_confidence_scores(target_data, prior_distribution=None):
    # if prior_distribution is None:
    #     prior_distribution = target_data
    # transformer = DataTransformer(prior_distribution)
    # return transformer.mapping_function(target_data)
    return target_data[0]
