class ScoreCriterion:
    def __init__(self, min_episode, min_avg_score, min_test_score, min_avg_test_score):
        self.min_episode = min_episode
        self.min_avg_score = min_avg_score
        self.min_test_score = min_test_score
        self.min_avg_test_score = min_avg_test_score

    def criterion(self, avg_score, test_score, avg_test_score, episode):
        if episode < self.min_episode:
            return False
        if avg_score < self.min_avg_score:
            return False
        if test_score < self.min_test_score:
            return False
        if avg_test_score is None:
            return True
        if avg_test_score < self.min_avg_test_score:
            return False
        return True
