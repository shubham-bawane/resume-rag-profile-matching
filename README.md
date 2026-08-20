# Resume RAG Profile Matching

This project implements a lightweight resume matching system using:

- document chunking
- metadata extraction
- embedding generation
- vector search with ChromaDB
- job-to-resume matching with semantic and keyword relevance

## Project Structure

- `resume_rag.py` — builds the vector database from resume files
- `job_matcher.py` — matches a job description against stored resumes
- `job_description.txt` — sample job description
- `resumes/` — folder containing resume files
- `requirements.txt` — project dependencies

## Features

- Loads resumes from `.txt`, `.pdf`, and `.docx` files
- Splits text into meaningful chunks while preserving sections like Education and Experience
- Extracts candidate metadata such as name, skills, experience years, and education
- Stores embeddings in a local Chroma collection
- Queries the job description against indexed resumes
- Returns top matches with score, relevant skills, excerpts, and reasoning

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the indexing pipeline

```bash
python resume_rag.py --resume-dir resumes --persist-dir vector_store
```

## Run the matching pipeline

```bash
python job_matcher.py --job-file job_description.txt --persist-dir vector_store --top-k 10
```

You can also pass the job description directly:

```bash
python job_matcher.py --job-description "Senior Python Data Scientist with 5+ years of experience" --persist-dir vector_store --top-k 10
```

## Example Output

```json
{
  "job_description": "...",
  "top_matches": [
    {
      "candidate_name": "John Doe",
      "resume_path": "resumes/john_doe.pdf",
      "match_score": 92,
      "matched_skills": ["Python", "Machine Learning"],
      "relevant_excerpts": ["..."],
      "reasoning": "Strong match for ML experience..."
    }
  ]
}
```

## Google Colab Usage

Install dependencies in Colab:

```python
!pip install -q chromadb sentence-transformers pypdf python-docx
```

Then run:

```python
!python resume_rag.py --resume-dir resumes --persist-dir vector_store
!python job_matcher.py --job-file job_description.txt --persist-dir vector_store --top-k 10
```

## Notes

- The project is designed for learning and assignment use.
- For production work, you may want to add stronger resume parsing, better metadata extraction, and a more advanced hybrid ranking strategy.
