# AskMyPDF

A Generative AI project (RAG): upload a PDF, ask a question in plain language, and get an answer with the page number for each fact. If the answer is not in the PDF, the app says so instead of guessing.

## How it works

1. The PDF text is split into overlapping chunks of about 800 characters, and each chunk keeps its page number.
2. For a question, two searches run: a meaning search (sentence-transformer embeddings) and an exact-word search (BM25). Their results are merged by rank (hybrid search).
3. The 12 best chunks go to Gemini, with an instruction to answer only from them and to cite pages.
4. If a Gemini model is busy, the app tries the next one.

## Testing

I tested it on the free book *Think Python* (2nd edition) by Allen B. Downey, about 250 pages. The PDF is not included here; the official page is https://greenteapress.com/thinkpython2

The test has 20 questions in `questions.csv`: 16 whose answers are in the book and 4 that are not. A question counts as correct when the answer contains the expected key word. Out-of-book questions count as correct when the app says it could not find the answer. `eval.py` runs the test.

| Run | What changed | Book questions correct | Out-of-book refused |
|---|---|---|---|
| 1 | Meaning search only, 6 chunks | 12 of 16 | 3 of 4 |
| 2 | Added exact-word search (hybrid) | 14 of 16 | 4 of 4 |
| 3 | Sent 12 chunks instead of 6 | 15 of 16 | 4 of 4 |

What I found: in run 1, four failures were search misses (the answer was in the book but not retrieved), and one out-of-book question was a mistake of mine, because the book does mention the person. I replaced that question and a weak one after run 1, so the test was tuned a little against these questions.

## Limitations

- Small test written by me: 20 questions, one book.
- The key-word check is simple. A correct answer in different words can be marked wrong.
- One question still fails: "What does the book say a program is?". The app answers "I could not find this in the document", even though the book defines the term. I did not tune the app further, to keep the test result honest.
- Scanned PDFs without a text layer are not supported.

## How to run

1. `pip install -r requirements.txt`
2. Create `key.txt` with your own Gemini API key on one line.
3. Start the app: `python -m streamlit run app.py`
4. To run the test, download the book, save it as `thinkpython.pdf` in the folder, and run `python eval.py`.

## Tech

Python, PyMuPDF, sentence-transformers, rank-bm25, Google Gemini API, Streamlit.

## Author

Shraddha Nimbalkar
