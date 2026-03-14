# 💭 Reflection: Game Glitch Investigator

Answer each question in 3 to 5 sentences. Be specific and honest about what actually happened while you worked. This is about your process, not trying to sound perfect.

## 1. What was broken when you started?

- What did the game look like the first time you ran it?
- List at least two concrete bugs you noticed at the start  
  (for example: "the hints were backwards").

One bug that I noticed is that you can guess a negative number when it said to guess between 1-100. I guessed 2 and it said to go lower, so I put a negative number and it told me to go lower still.
Another bug that I noticed is that the history duplicates your guess once before it puts your next guess. ex: i put 3 first then 100, in the history it says 3 3 100.
Third thing I noticed is that the new game button is broken. I pressed it and the attempts changes, but submitting a guess wouldn't work because it would say "Game over. Start A new game to try again"

## 2. How did you use AI as a teammate?

- Which AI tools did you use on this project (for example: ChatGPT, Gemini, Copilot)?
- Give one example of an AI suggestion that was correct (including what the AI suggested and how you verified the result).
- Give one example of an AI suggestion that was incorrect or misleading (including what the AI suggested and how you verified the result).

I used Claude for this project. When I told it to fix the code based on my fix me comments, it did what it was told to but it doesn't actually test for the test cases that might break it. When I told it to write a pytest to test out the bugs that it fixed, it started to search through the code and realized it's mistake and thats when it went back and fixed everything that it missed. 

## 3. Debugging and testing your fixes

- How did you decide whether a bug was really fixed?
- Describe at least one test you ran (manual or using pytest)  
  and what it showed you about your code.
- Did AI help you design or understand any tests? How?

I decided if a bug was really fixed when I went to test the bugs manual to see if it was actually fixed because I could tell if it did or not. For one of test I tested numbers out of bound of what we are supposed to guess from and when I tested it thats when I knew Claude did what I asked. Since it would output how you need to guess in between the high and low and it wouldn't take off an attempt from guesses. Yes AI did help any test since it had comments in the test code and reading through it makes me understand what edge cases I need to test/ I wouldn't even think of. 

## 4. What did you learn about Streamlit and state?

- How would you explain Streamlit "reruns" and session state to a friend who has never used Streamlit?

---

## 5. Looking ahead: your developer habits

- What is one habit or strategy from this project that you want to reuse in future labs or projects?
  - This could be a testing habit, a prompting strategy, or a way you used Git.
- What is one thing you would do differently next time you work with AI on a coding task?
- In one or two sentences, describe how this project changed the way you think about AI generated code.
