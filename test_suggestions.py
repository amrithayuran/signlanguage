from app import SuggestionEngine
import config

def test_suggestions():
    print("Testing Suggestion Engine...")
    try:
        engine = SuggestionEngine()
        
        # Test 1: Check wordlist size
        print(f"Wordlist size: {len(engine.wordlist)}")
        if len(engine.wordlist) < 1000:
            print("WARNING: Wordlist seems too small!")
        else:
            print("Wordlist size looks good.")

        # Test 2: Get suggestions for a prefix
        prefix = "he"
        suggestions = engine.get_suggestions(prefix)
        print(f"Suggestions for '{prefix}': {suggestions}")
        
        if not suggestions:
            print("FAIL: No suggestions returned.")
        else:
            print("PASS: Suggestions returned.")

        # Test 3: Check if common words are present
        common_words = ['hello', 'world', 'beautiful', 'friend']
        missing = [w for w in common_words if w not in engine.wordlist]
        if missing:
            print(f"WARNING: Missing common words: {missing}")
        else:
            print("PASS: Common words found in wordlist.")

    except Exception as e:
        print(f"Suggestion Engine Test Failed: {e}")

if __name__ == "__main__":
    test_suggestions()
