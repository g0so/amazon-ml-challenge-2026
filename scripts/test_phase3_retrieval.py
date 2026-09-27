import unittest
import math

# 1. Oracle F0.5 Implementation
def oracle_f05(retrieved_candidates, true_targets):
    """Calculates Oracle F0.5 where a hypothetical classifier perfectly filters candidates."""
    g = len(true_targets)
    if g == 0:
        return 1.0  # Classifier perfectly rejects all candidates; score is 1.
    
    t = len(retrieved_candidates & true_targets)
    if t == 0:
        return 0.0  # Missed all true targets
    
    return (1.25 * t) / (t + 0.25 * g)

# 2. Tokenization Implementation
import re
def tokenize(text):
    """Unicode-aware tokenization preserving letters and digits, deduplicated."""
    if not text:
        return set()
    # [^\W_]+ matches 1+ unicode word characters excluding underscore
    return set(re.findall(r'[^\W_]+', text.lower()))

class TestPhase3Retrieval(unittest.TestCase):
    def test_oracle_f05(self):
        # g=0 gives 1 (Singleton perfectly handled by oracle)
        self.assertEqual(oracle_f05({'S2-999'}, set()), 1.0)
        self.assertEqual(oracle_f05(set(), set()), 1.0)
        
        # g>0, t=0 gives 0
        self.assertEqual(oracle_f05({'S2-999'}, {'S2-111'}), 0.0)
        self.assertEqual(oracle_f05(set(), {'S2-111'}), 0.0)
        
        # 2 of 4 true targets retrieved: 1.25(2) / (2 + 0.25(4)) = 2.5 / 3 = 0.8333...
        score = oracle_f05({'S2-1', 'S2-2', 'S3-99'}, {'S2-1', 'S2-2', 'S2-3', 'S2-4'})
        self.assertAlmostEqual(score, 0.8333333333)

    def test_tokenization_and_deduplication(self):
        # Punctuation removal, case folding, unicode preservation, deduplication
        tokens = tokenize("Café Alpha, 12! CAFE café")
        self.assertEqual(tokens, {"café", "alpha", "12", "cafe"})
        
        # Empty field
        self.assertEqual(tokenize("   ,,,   "), set())
        self.assertEqual(tokenize(None), set())

    def test_deterministic_sorting(self):
        # Tied query tokens: Sort by (-IDF, Token ASC)
        tokens = [("token_b", 5.0), ("token_a", 5.0), ("token_c", 2.0)]
        sorted_tokens = sorted(tokens, key=lambda x: (-x[1], x[0]))
        self.assertEqual(sorted_tokens, [("token_a", 5.0), ("token_b", 5.0), ("token_c", 2.0)])
        
        # Tied target scores: Sort by (-Score, Target Int ID ASC)
        targets = [(105, 12.5), (99, 12.5), (200, 8.0)]
        sorted_targets = sorted(targets, key=lambda x: (-x[1], x[0]))
        self.assertEqual(sorted_targets, [(99, 12.5), (105, 12.5), (200, 8.0)])

if __name__ == '__main__':
    unittest.main()