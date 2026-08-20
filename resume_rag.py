import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


SECTION_HEADERS = {
    "education": ["education", "academics", "qualification", "qualifications"],
    "experience": ["experience", "work experience", "professional experience", "employment"],
    "skills": ["skills", "technical skills", "core competencies", "tools", "technologies"],
    "projects": ["projects", "project work", "notable projects"],
    "summary": ["summary", "profile", "about me", "overview"],
    "certifications": ["certifications", "licenses", "training"],
}

COMMON_SKILLS = {
    "python",
    "sql",
    "machine learning",
    "deep learning",
    "nlp",
    "pandas",
    "numpy",
    "scikit-learn",
    "pytorch",
    "tensorflow",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "java",
    "javascript",
    "c++",
    "react",
    "nodejs",
    "mongodb",
    "postgresql",
    "spark",
    "hadoop",
    "tableau",
    "power bi",
    "airflow",
    "data engineering",
    "llm",
    "rag",
    "langchain",
    "vector database",
    "chroma",
    "faiss",
    "linux",
    "api",
    "rest",
    "etl",
    "html",
    "css",
    "git",
    "database",
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_text_from_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    if suffix == ".docx":
        doc = Document(str(file_path))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
    return ""


def detect_section_name(line: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z\s/\-&]", "", line.strip().lower())
    for standard, aliases in SECTION_HEADERS.items():
        for alias in aliases:
            if cleaned == alias or cleaned.startswith(alias + " "):
                return standard
    return "general"


def chunk_resume_text(resume_text: str) -> List[Tuple[str, str]]:
    lines = [line.strip() for line in resume_text.splitlines()]
    chunks: List[Tuple[str, str]] = []
    current_section = "general"
    current_lines: List[str] = []

    def flush_current():
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                chunks.append((current_section, content))

    for line in lines:
        if not line:
            if current_lines:
                current_lines.append("")
            continue
        section = detect_section_name(line)
        if section != "general":
            flush_current()
            current_section = section
            current_lines = [line]
        else:
            current_lines.append(line)

    flush_current()

    final_chunks: List[Tuple[str, str]] = []
    for section, content in chunks:
        paragraphs = re.split(r"\n\s*\n+", content)
        for para in paragraphs:
            clean_para = re.sub(r"\s+", " ", para).strip()
            if not clean_para:
                continue
            if len(clean_para) <= 500:
                final_chunks.append((section, clean_para))
            else:
                sentences = re.split(r"(?<=[.!?])\s+", clean_para)
                grouped = []
                temp = ""
                for sentence in sentences:
                    if len(temp) + len(sentence) <= 450:
                        temp = f"{temp} {sentence}".strip()
                    else:
                        if temp:
                            grouped.append(temp)
                        temp = sentence
                if temp:
                    grouped.append(temp)
                for item in grouped:
                    final_chunks.append((section, item))
    return final_chunks or [("general", normalize_space(resume_text)[:1500])]


def extract_name_from_resume(text: str) -> str:
    candidates = [
        re.search(r"\bName\s*[:\-]?\s*([A-Z][A-Za-z' .-]+)", text, re.I),
        re.search(r"^\s*([A-Z][A-Za-z' .-]+)\s*$", text, re.M),
    ]
    for match in candidates:
        if match:
            return match.group(1).strip()
    return "Unknown Candidate"


def extract_skills_from_resume(text: str) -> List[str]:
    lower_text = text.lower()
    found = []
    for skill in sorted(COMMON_SKILLS, key=len, reverse=True):
        if skill in lower_text:
            found.append(skill)
    return found[:15]


def extract_experience_years(text: str) -> float:
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s+)?experience",
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:in\s+)?(?:data|software|ai|ml|product|engineering)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*yrs?\s*(?:of\s+)?experience",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1))
    return 0.0


def extract_education(text: str) -> str:
    education_patterns = [
        r"(?:B\.?Tech|BTech|Bachelor(?:s)?|M\.?Tech|MTech|Master(?:s)?|MBA|B\.Sc|M\.Sc|PhD|Diploma)[^\n]*",
        r"(?:Computer Science|Information Technology|Engineering|Business Administration)[^\n]*",
    ]
    for pattern in education_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(0).strip()
    return "Not specified"


def build_resume_index(resume_dir: str = "resumes", persist_dir: str = "vector_store") -> Dict[str, object]:
    resume_path = Path(resume_dir)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume folder does not exist: {resume_dir}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name="resume_collection")

    all_ids: List[str] = []
    all_documents: List[str] = []
    all_embeddings: List[List[float]] = []
    all_metadatas: List[Dict[str, str]] = []

    for file_path in sorted(resume_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".txt", ".pdf", ".docx"}:
            continue
        text = extract_text_from_file(file_path)
        if not text.strip():
            continue

        candidate_name = extract_name_from_resume(text) or file_path.stem
        skills = extract_skills_from_resume(text)
        experience_years = extract_experience_years(text)
        education = extract_education(text)
        chunks = chunk_resume_text(text)

        for idx, (section, chunk) in enumerate(chunks):
            chunk_id = f"{file_path.stem}-{idx}"
            all_ids.append(chunk_id)
            all_documents.append(chunk)
            all_embeddings.extend(model.encode([chunk]).tolist())
            all_metadatas.append(
                {
                    "candidate_name": candidate_name,
                    "resume_path": str(file_path),
                    "section": section,
                    "skills": json.dumps(skills),
                    "experience_years": str(experience_years),
                    "education": education,
                }
            )

    if all_documents:
        collection.add(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_documents,
            metadatas=all_metadatas,
        )

    return {
        "collection_name": "resume_collection",
        "persist_dir": persist_dir,
        "resume_count": len({meta["resume_path"] for meta in all_metadatas}),
        "chunk_count": len(all_documents),
    }


def main():
    parser = argparse.ArgumentParser(description="Index resumes into a local Chroma vector store.")
    parser.add_argument("--resume-dir", default="resumes", help="Folder containing resumes")
    parser.add_argument("--persist-dir", default="vector_store", help="Directory for the Chroma database")
    args = parser.parse_args()

    result = build_resume_index(args.resume_dir, args.persist_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
