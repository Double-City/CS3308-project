import numpy as np

class TSNE:
    def __init__(self, output_dims=2, perplexity=30.0, max_iterations=200, learning_rate=500):
        self.output_dims = output_dims
        self.perplexity = perplexity
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    @staticmethod
    def calculate_perplexity(distance, index=0, beta=1.0):
        prob = np.exp(-distance * beta)
        prob[index] = 0
        total_prob = np.sum(prob)

        if total_prob == 0:
            prob = np.maximum(prob, 1e-12)
            perplexity = -12
        else:
            prob /= total_prob
            perplexity = -np.sum(prob * np.log(prob + 1e-12))

        return perplexity, prob

    def search_prob(self, data, tolerance=1e-5):
        (n, d) = data.shape
        squared_sum = np.sum(np.square(data), 1)
        distance = np.add(np.add(-2 * np.dot(data, data.T), squared_sum).T, squared_sum)
        pairwise_prob = np.zeros((n, n))
        beta = np.ones((n, 1))

        log_perplexity = np.log(self.perplexity)

        for i in range(n):
            beta_min = -np.inf
            beta_max = np.inf

            perplexity, current_prob = self.calculate_perplexity(distance[i], i, beta[i])

            perplexity_diff = perplexity - log_perplexity
            j = 0
            while np.abs(perplexity_diff) > tolerance and j < 50:
                if perplexity_diff > 0:
                    beta_min = beta[i]
                    if (beta_max == np.inf or beta_max == -np.inf):
                        beta[i] = beta[i] * 2
                    else:
                        (beta[i] + beta_max) / 2
                else:
                    beta_max = beta[i]
                    if (beta_min == np.inf or beta_min == -np.inf):
                        beta[i] = beta[i] / 2
                    else:
                        (beta[i] + beta_min) / 2

                perplexity, current_prob = self.calculate_perplexity(distance[i], i, beta[i])
                perplexity_diff = perplexity - log_perplexity
                j += 1

            pairwise_prob[i] = current_prob

        return pairwise_prob

    def fit_transform(self, data):
        print("Starting t-SNE")

        n, _ = data.shape
        y = np.random.randn(n, self.output_dims)
        gradient = np.zeros((n, self.output_dims))

        P = self.search_prob(data, 1e-5)
        P += np.transpose(P)
        P /= np.sum(P)
        P = np.maximum(P * 4, 1e-12)

        for iteration in range(self.max_iterations):
            sum_y = np.sum(np.square(y), 1)
            num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
            num[range(n), range(n)] = 0
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            PQ_diff = P - Q

            for i in range(n):
                gradient[i,:] = np.sum(np.tile(PQ_diff[:,i] * num[:,i], (self.output_dims, 1)).T * (y[i,:] - y), 0)
            y -= self.learning_rate*gradient
            y -= np.mean(y, axis=0)

        print("Finished t-SNE!")
        return y