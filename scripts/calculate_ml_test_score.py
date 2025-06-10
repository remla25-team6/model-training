import subprocess
import re

def get_test_names():
    # Used ChatGPT to get the command below which gets the names of all pytest cases.
    result = subprocess.run(
        ['pytest', '--collect-only', '-q'],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

def score_from_test_names(test_names):
    categories = ["data", "model", "infra", "monitor"]
    counts = {category: set() for category in categories}
    for test_name in test_names:
        if "::" not in test_name:
            continue
        func_name = test_name.split("::")[-1]

        # Used ChatGPT to generate the pattern matching
        match = re.match(r"test_(data|model|infra|monitor)_(\d+)_", func_name)
        if match:
            category, number = match.groups()
            counts[category].add(number)

    counts = {category: len(numbers) for category, numbers in counts.items()}

    return min(counts.values())

if __name__ == "__main__":
    tests = get_test_names()
    score = score_from_test_names(tests)
    print(score)
