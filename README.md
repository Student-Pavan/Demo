

# ⚡ Quiz Generator — Text, Images & PDFs

Generate MCQs with difficulty levels from any content using **free HuggingFace models**.

## Features

- 📝 **Text** — paste any article, lesson, or notes
- 🖼️ **Images** — OCR-powered question generation from photos
- 📄 **PDF** — reads embedded text + OCR fallback for scanned pages
- 🎯 **Difficulty Filter** — filter questions by Easy / Medium / Hard
- ✅ **Instant Feedback** — see right/wrong + correct answer after submitting
- 📊 **Score Summary** — score card at the end

## Free Models Used (no API key needed)

| Purpose | Model |
|---|---|
| Question Generation | `google/flan-t5-base` |
| OCR (images/scanned PDFs) | `microsoft/trocr-base-printed` |
| Difficulty Classification | TF-IDF + Logistic Regression (sklearn, local) |

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## Deploy to Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space**
2. Choose **Gradio** SDK
3. Upload all files in this folder
4. HuggingFace will auto-install `requirements.txt` and run `app.py`

> ⏳ First launch downloads models (~1–2 GB). Subsequent launches use cache.

## Optional: Difficulty Classifier

If `models/difficulty_model.pkl` is present, questions get difficulty badges.
To train:

```bash
python train_difficulty_model.py
# or with custom CSV (columns: question, label):
python train_difficulty_model.py --csv your_data.csv
```

## Notes

- Works fully on CPU — no GPU required (GPU speeds it up significantly)
- HuggingFace Spaces free tier works; use **T4 GPU** hardware for faster generation
- PDF extraction reads native text first and falls back to OCR for scanned pages
