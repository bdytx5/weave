"""Quick test script for LiteEvaluationLogger"""
import weave
from weave.evaluation.eval_imperative import LiteEvaluationLogger

weave.init("lite-logger-test")


# Dummy model function
@weave.op
def dummy_qa_model(question: str) -> dict:
    """Simple dummy Q&A model that gives canned responses"""
    responses = {
        "What is 2+2?": "4",
        "What is the capital of France?": "Paris",
        "What color is the sky?": "Blue",
    }
    return {
        "answer": responses.get(question, "I don't know"),
        "confidence": 0.95 if question in responses else 0.3
    }


# Dummy scorer function
def is_correct(output: dict, expected: str) -> bool:
    """Check if the answer matches expected"""
    return output.get("answer", "").lower() == expected.lower()


def main():
    # Initialize weave
    weave.init("lite-logger-test")

    # Create dummy dataset
    test_questions = [
        {"question": "What is 2+2?", "expected_answer": "4"},
        {"question": "What is the capital of France?", "expected_answer": "Paris"},
        {"question": "What color is the sky?", "expected_answer": "Blue"},
        {"question": "What is quantum physics?", "expected_answer": "Complex science"},
    ]

    # Create logger
    logger = LiteEvaluationLogger(
        model="dummy_qa_model_v1",
        dataset=test_questions,
        name="test_evaluation"
    )

    print("Running evaluation with LiteEvaluationLogger...")
    print("-" * 50)

    # Evaluate each example
    for idx, example in enumerate(test_questions, 1):
        # Get model output
        output = dummy_qa_model(example["question"])

        # Calculate scores
        correct = is_correct(output, example["expected_answer"])
        confidence = output["confidence"]

        # Log everything in one call
        logger.log_example(
            inputs={"question": example["question"]},
            output=output,
            scores={
                "correct": correct,
                "confidence": confidence,
                "latency_ms": 50 + idx * 10  # fake latency
            }
        )

        status = "✓" if correct else "✗"
        print(f"{status} Example {idx}: {example['question'][:40]}... -> {output['answer']}")

    print("-" * 50)
    print("All examples logged!")
    print("\nNote: log_summary() will be called automatically at exit")
    print("Or you can call logger.log_summary() manually now")

    # Optional: Call log_summary manually
    # logger.finish()
    print("Summary logged!")


if __name__ == "__main__":
    main()