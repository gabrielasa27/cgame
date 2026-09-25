import unittest

from recommender.levels import LEVELS, LEVEL_ORDER
from recommender.model import LinearSVM
from recommender.recommend import recommend
from recommender.scoring import compute_result
from recommender.simulate import simulate_students


def attempt(level_id, **kw):
    lv = LEVELS[level_id]
    base = dict(level_id=level_id, game_id=lv.game_id, completed=True,
                accuracy=95, movements=lv.optimal_moves, time=lv.par_time, errors=0,
                hints_used=0, score=95)
    base.update(kw)
    return base


class LevelsTest(unittest.TestCase):
    def test_optimal_moves(self):
        got = {k: v.optimal_moves for k, v in LEVELS.items()}
        self.assertEqual(got, {"G1-L1": 5, "G1-L2": 8, "G1-L3": 12, "G2-L1": 6, "G2-L2": 10,
                               "G2-L3": 16, "G3-L1": 4, "G3-L2": 5, "G3-L3": 6})

    def test_slots_allow_optimal(self):
        for lv in LEVELS.values():
            self.assertGreaterEqual(lv.max_slots, lv.optimal_moves, lv.id)


class ScoringTest(unittest.TestCase):
    def test_perfect_attempt(self):
        r = compute_result(LEVELS["G1-L1"], hits=5, misses=0, movements=5, time_s=20, hints_used=0)
        self.assertEqual((r["score"], r["accuracy"], r["approved"]), (100, 100, True))

    def test_hints_penalize(self):
        r = compute_result(LEVELS["G1-L1"], 5, 0, 5, 20, hints_used=2)
        self.assertEqual(r["score"], 90)

    def test_poor_attempt_not_approved(self):
        r = compute_result(LEVELS["G1-L2"], hits=8, misses=12, movements=20, time_s=200, hints_used=1)
        self.assertFalse(r["approved"])

    def test_incomplete_never_approved(self):
        r = compute_result(LEVELS["G1-L1"], 5, 0, 5, 20, 0, completed=False)
        self.assertFalse(r["approved"])


class RecommendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LinearSVM.load()

    def test_good_attempt_advances(self):
        rec = recommend(self.model, [attempt("G1-L1")])
        self.assertEqual((rec["action"], rec["level_id"], rec["prediction"]), ("advance", "G1-L2", "aprendio"))

    def test_good_attempt_last_level_moves_to_next_pending_game(self):
        hist = [attempt("G1-L1"), attempt("G1-L2"), attempt("G1-L3")]
        rec = recommend(self.model, hist)
        self.assertEqual(rec["level_id"], "G2-L1")

    def test_weak_attempt_reinforces_same_level(self):
        weak = attempt("G1-L2", accuracy=35, movements=20, time=200, errors=6, hints_used=3, score=30)
        rec = recommend(self.model, [weak])
        self.assertEqual((rec["action"], rec["level_id"]), ("reinforce", "G1-L2"))

    def test_two_failures_step_back(self):
        weak = attempt("G1-L2", accuracy=35, movements=20, time=200, errors=6, hints_used=3, score=30)
        rec = recommend(self.model, [weak, dict(weak)])
        self.assertEqual((rec["action"], rec["level_id"]), ("step_back", "G1-L1"))

    def test_incomplete_attempts_ignored(self):
        self.assertIsNone(recommend(self.model, [attempt("G1-L1", completed=False)]))

    def test_all_done_reviews_weakest(self):
        hist = [attempt(lid) for lid in LEVEL_ORDER]
        hist[3]["score"] = 65  # G2-L1 es el mejor "peor" resultado
        rec = recommend(self.model, hist)
        self.assertEqual(rec["action"], "review")
        self.assertEqual(rec["level_id"], "G2-L1")


class SimulationTest(unittest.TestCase):
    def test_students_are_reproducible_and_have_all_fields(self):
        a, b = simulate_students(3, 5), simulate_students(3, 5)
        self.assertEqual(a, b)
        for key in ("user_id", "game_id", "level_id", "attempt_id", "score", "accuracy",
                    "time", "movements", "errors", "hints_used", "completed", "date"):
            self.assertIn(key, a[0])


if __name__ == "__main__":
    unittest.main()
