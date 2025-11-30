from src.validator import AtomicAnswerValidator
import json

def run_demo():
    # 1. Initialize
    system = AtomicAnswerValidator()

    # 2. Define Data (Photosynthesis Example)
    context = """
    Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy. 
    This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water. 
    The process takes place in chloroplasts, which contain chlorophyll. 
    Oxygen is released as a byproduct of this reaction.
    """
    
    question = "What is photosynthesis?"

    # Test Case A: A Correct Answer
    ans_correct = "Photosynthesis occurs in chloroplasts. It converts light energy into chemical energy."

    # Test Case B: A Subtle Hallucination (Logical Contradiction)
    # "Oxygen is consumed" contradicts "Oxygen is released"
    ans_hallucination = "Photosynthesis converts light energy into chemical energy. Oxygen is consumed in the process."

    print("\n--- TEST CASE A: Correct Answer ---")
    result_a = system.validate(question, context, ans_correct)
    print(json.dumps(result_a, indent=2))

    print("\n--- TEST CASE B: Hallucinated Answer ---")
    result_b = system.validate(question, context, ans_hallucination)
    print(json.dumps(result_b, indent=2))

if __name__ == "__main__":
    run_demo()
