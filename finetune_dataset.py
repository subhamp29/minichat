"""Build the curated fine-tuning dataset (finetune_dataset.jsonl).

Run:  python finetune_dataset.py
Out:  finetune_dataset.jsonl  (one JSON object per line)

Each record is a multi-turn chat:
    {"messages": [{"role": "system", "content": ...},
                  {"role": "user",   "content": ...},
                  {"role": "assistant", "content": ...}]}

The dataset is intentionally small and HIGH QUALITY, biased toward the formats
the base Qwen2.5-0.5B model gets wrong today (arithmetic, comparisons, code,
factual reasoning). Correct, concise, faithful answers are the whole point -
we are training accuracy, not verbosity.
"""
import json

SYSTEM = (
    "You are Bhavyam AI, an accurate and helpful assistant created and owned by "
    "Subham Mahapatra from Odisha, India. Answer concisely and precisely. "
    "For math and programming questions, give correct results only. For factual "
    "questions, state verified facts. When a fact is genuinely outside your "
    "training knowledge, say briefly that you do not know rather than guessing."
)

# (user, assistant) pairs. Correctness is the label.
PAIRS = [
    # ---- arithmetic / numeric correctness (the most common failure mode) ----
    ("Is 5 greater than 3?", "True. 5 is greater than 3."),
    ("Is 5 less than 3?", "False. 5 is greater than 3, so 5 < 3 is false."),
    ("Is 5 < 3? State true or false.", "False. 5 is greater than 3."),
    ("Is 3 greater than 5?", "False. 3 is less than 5."),
    ("What is 2 + 2?", "4."),
    ("What is 8 times 7?", "56."),
    ("What is 100 minus 37?", "63."),
    ("What is 9 times 9?", "81."),
    ("What is 12 divided by 4?", "3."),
    ("What is 7 * 6?", "42."),
    ("What is 2 + 2 * 3?", "8. Multiplication before addition: 2 + 6 = 8."),
    ("What is the sum of 1 to 100?",
     "5050. The sum 1+2+...+100 = 100*101/2 = 5050."),
    ("Is 0.5 greater than 0.25?", "True. 0.5 is greater than 0.25."),
    ("Is 1 greater than 2?", "False. 1 is less than 2."),
    ("Compare 99 and 100. Which is larger?", "100 is larger than 99."),
    ("Which is bigger, 7 or 8?", "8 is bigger than 7."),
    ("What is 2 to the power of 10?", "1024."),
    ("What is the remainder when 17 is divided by 5?", "2. 17 = 3*5 + 2."),
    ("Is 10 percent greater than 0.09?", "True. 10 percent = 0.10, which is greater than 0.09."),
    ("Order these from smallest to largest: 3, 1, 2.", "1, 2, 3."),

    # ---- algebra ----
    ("Solve 2x + 5 = 11 for x.", "x = 3. Subtract 5: 2x = 6, divide by 2: x = 3."),
    ("Solve x^2 = 16 for x.", "x = 4 or x = -4."),
    ("What is the value of x if 3x = 21?", "x = 7."),
    ("Factor x^2 - 5x + 6.", "(x - 2)(x - 3)."),
    ("What is (a + b)^2?", "a^2 + 2ab + b^2."),
    ("Solve y = 2x + 3 when x = 4.", "y = 11. 2*4 + 3 = 11."),

    # ---- calculus / simple reasoning ----
    ("What is the derivative of x^2?", "2x."),
    ("What is the derivative of 3x^3?", "9x^2."),
    ("What is the integral of 2x?", "x^2 + C."),
    ("What is sin^2(x) + cos^2(x)?", "1. This is the Pythagorean identity."),
    ("Evaluate the limit of (sin x)/x as x approaches 0.", "1."),
    ("What is the slope of a horizontal line?", "0."),
    ("What is the area of a circle with radius r?", "pi * r^2."),

    # ---- programming: correctness ----
    ("In Python, what is the output of `print(2 + 2)`?",
     "4. It prints 4."),
    ("Is `==` the same as `===` in Python?",
     "No. Python only has `==` for equality; there is no `===` operator."),
    ("What does `len('hello')` return in Python?", "5."),
    ("What is a list comprehension in Python?",
     "A concise way to build a list, e.g. `[x for x in range(3)]` gives [0, 1, 2]."),
    ("How do you reverse a list in Python?",
     "Several ways: `lst.reverse()` (in place), `lst[::-1]` (new list), or `reversed(lst)`."),
    ("What is the index of 'c' in 'abc'?", "0-based: a=0, b=1, c=2. The index is 2."),
    ("In Python, is `a = 5` the same as `a == 5`?",
     "No. `=` assigns, `==` compares."),
    ("Which language: HTML is a markup language or a programming language?",
     "A markup language, not a programming language."),
    ("What does CSS stand for?",
     "Cascading Style Sheets."),
    ("JavaScript: what does `===` check?",
     "Strict equality: same type and value (e.g. 5 === '5' is false)."),
    ("What is null in JavaScript?",
     "An assignment value representing no value (type object, value null)."),

    # ---- simple code writing (correctness focused) ----
    ("Write a Python function that returns the larger of two numbers.",
     "def larger(a, b):\n    return a if a > b else b"),
    ("Write a Python function to check if a number is even.",
     "def is_even(n):\n    return n % 2 == 0"),
    ("Write a Python loop that prints numbers 0 to 4.",
     "for i in range(5):\n    print(i)"),

    # ---- factual (entities, dates, science) ----
    ("Who is Narendra Modi?",
     "Narendra Modi is the current Prime Minister of India, in office since 2014."),
    ("Who is the current Prime Minister of India?",
     "Narendra Modi, since 2014."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital city of France?", "Paris."),
    ("What is the capital of India?", "New Delhi."),
    ("What is the capital of Germany?", "Berlin."),
    ("When was the Eiffel Tower completed?",
     "It was completed in 1889."),
    ("When was the Eiffel Tower built?",
     "Construction ran 1887-1889; it was completed in 1889."),
    ("How many continents are there?",
     "Seven: Asia, Africa, North America, South America, Antarctica, Europe, Australia."),
    ("What is the largest planet in our solar system?", "Jupiter."),
    ("What is the chemical symbol for water?", "H2O."),
    ("What is the chemical symbol for gold?", "Au."),
    ("What is the speed of light in a vacuum?",
     "Approximately 299,792,458 meters per second."),
    ("What is the boiling point of water?",
     "100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."),
    ("How many bones are in an adult human body?", "206."),
    ("What year did World War II end?", "1945."),
    ("Who wrote 'Romeo and Juliet'?", "William Shakespeare."),
    ("What planet is known as the Red Planet?", "Mars."),
    ("What is the currency of Japan?", "The yen (JPY)."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
    ("What is the tallest mountain above sea level?", "Mount Everest."),
    ("When was the first iPhone released?", "In 2007."),
    ("What is the smallest prime number?", "2."),
    ("Is Pluto a planet?", "No. It is classified as a dwarf planet."),
    ("Who formulated the theory of relativity?", "Albert Einstein."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean."),
    ("How many days are in a leap year?", "366."),
    ("What element has the atomic number 1?", "Hydrogen."),
    ("What is the capital of Australia?",
     "Canberra (not Sydney or Melbourne)."),

    # ---- conceptual / definitional ----
    ("Explain gravity in one sentence.",
     "Gravity is the force that attracts objects toward one another; on Earth it gives weight to physical bodies."),
    ("What is machine learning?",
     "A field of AI where computers learn patterns from data rather than explicit programming."),
    ("What is photosynthesis?",
     "The process by which green plants convert light energy, water, and carbon dioxide into oxygen and glucose."),
    ("What is the difference between weather and climate?",
     "Weather is short-term atmospheric conditions; climate is the long-term average of weather over years."),
    ("Define HTTP.",
     "Hypertext Transfer Protocol, the foundation of data communication on the web."),

    # ---- reasoning / logic (true-false shaped) ----
    ("True or false: all squares are rectangles.",
     "True. Every square is a rectangle (it has four right angles), but not every rectangle is a square."),
    ("True or false: a month can have 28 days.",
     "True. February has 28 days (29 in a leap year)."),
    ("Is 0 an even number?", "Yes, 0 is an even number."),
    ("True or false: 5 < 3.", "False. 5 is greater than 3."),
    ("Is the statement 'all birds can fly' true?",
     "False. Some birds, like penguins and ostriches, cannot fly."),
    ("If today is Monday, what day is it in 72 hours?",
     "Thursday. 72 hours = 3 days; Monday + 3 days = Thursday."),
    ("What is 50 minus 50?", "0."),

    # ---- honesty on unknowns (prevents hallucination) ----
    ("When exactly will the next solar eclipse be visible from my city?",
     "I don't have your city or a precise future-prediction source; check a local astronomy almanac for the next solar eclipse near you."),
    ("What was the exact stock price of my private company on 1 January 2025?",
     "I don't have access to real-time or private market data."),
    ("Predict cryptocurrency prices for next month.",
     "I can't reliably predict future prices; that is speculation, not knowledge."),
    ("What is the winning lottery number for tomorrow?", "I cannot know or predict lottery numbers."),
]


def main():
    out_path = "finetune_dataset.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for user, answer in PAIRS:
            obj = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} examples to {out_path}")


if __name__ == "__main__":
    main()
