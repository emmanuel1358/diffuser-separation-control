import math


def load_policy():
    class UCBPolicy:
        def reset(self, n_arms, budget):
            self.n_arms = int(n_arms)
            self.budget = int(budget)
            self.counts = [0] * self.n_arms
            self.totals = [0.0] * self.n_arms
            self.steps = 0

        def choose(self):
            for arm, count in enumerate(self.counts):
                if count == 0:
                    return arm
            log_t = math.log(max(2, self.steps))
            return max(
                range(self.n_arms),
                key=lambda arm: (
                    self.totals[arm] / self.counts[arm]
                    + math.sqrt(2.0 * log_t / self.counts[arm])
                ),
            )

        def observe(self, arm, reward):
            arm = int(arm)
            self.counts[arm] += 1
            self.totals[arm] += float(reward)
            self.steps += 1

        def recommend(self):
            return max(
                range(self.n_arms),
                key=lambda arm: self.totals[arm] / max(1, self.counts[arm]),
            )

    return UCBPolicy()
