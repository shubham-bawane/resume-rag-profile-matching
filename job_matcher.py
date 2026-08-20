import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer


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

SECTION_WEIGHT = {
    "experience": 1.2,
    "skills": 1.15,
    "projects": 1.1,
    "summary": 0.9,
    "general": 0.8,
}


def normalize_skill(skill: str) -> str:
    return re.sub(r"\s+", " ", skill.strip().lower())


def extract_job_skills(job_text: str) -> List[str]:
    found = []
    lower_text = job_text.lower()
    for skill in sorted(COMMON_SKILLS, key=len, reverse=True):
        if skill in lower_text:
            found.append(skill)
    return found


def parse_must_have_requirements(job_text: str) -> List[Tuple[int, str]]:
    requirements: List[Tuple[int, str]] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s+)?([a-z0-9+\s/-]+)", job_text, re.I):
        years = float(match.group(1))
        skill = normalize_skill(match.group(2))
        if skill and skill not in {"experience", "professional", "software", "data"}:
            requirements.append((int(years), skill))
    return requirements


def calculate_similarity(distance: float) -> float:
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - float(distance))


def candidate_keyword_overlap(job_skills: List[str], text: str, metadata_skills: List[str]) -> float:
    candidate_text = " ".join(metadata_skills + [text]).lower()
    matches = [skill for skill in job_skills if skill in candidate_text]
    if not job_skills:
        return 0.0
    return len(matches) / len(job_skills)


def evaluate_candidate(candidate_records: List[Dict[str, object]], job_skills: List[str], must_haves: List[Tuple[int, str]]) -> Dict[str, object]:
    candidate_name = candidate_records[0]["candidate_name"]
    resume_path = candidate_records[0]["resume_path"]
    aggregated_skill_set = set()
    excerpts = []
    skill_hits = []
    relevant_sections = []
    best_score = 0.0

    for record in candidate_records:
        relevant_sections.append(record["section"])
        excerpts.append(record["excerpt"])
        skill_hit = [skill for skill in job_skills if skill in (record["text"] + " " + " ".join(record.get("metadata_skills", []))).lower()]
        skill_hits.extend(skill_hit)
        aggregated_skill_set.update(skill_hit)
        best_score = max(best_score, float(record["match_score"]))

    if job_skills:
        matched_skills = sorted(set(skill_hits), key=lambda x: job_skills.index(x) if x in job_skills else len(job_skills))
    else:
        matched_skills = []

    experience_years = float(candidate_records[0].get("experience_years", 0) or 0)
    passes_must_have = True
    for years_required, skill in must_haves:
        if skill in {"python", "sql", "aws", "java", "pytorch", "sql", "docker"}:
            if experience_years < years_required and skill not in aggregated_skill_set:
                passes_must_have = False
                break
        elif skill not in aggregated_skill_set:
            passes_must_have = False
            break

    return {
        "candidate_name": candidate_name,
        "resume_path": resume_path,
        "match_score": round(best_score, 2),
        "matched_skills": matched_skills[:10],
        "relevant_excerpts": excerpts[:3],
        "reasoning": "Strong job alignment across relevant resume sections." if best_score >= 75 else "Good overall fit with some relevant experience and skills.",
        "passes_must_have": passes_must_have,
        "experience_years": experience_years,
        "sections": relevant_sections,
    }


def match_job_to_resumes(job_description: str, persist_dir: str = "vector_store", top_k: int = 10) -> Dict[str, object]:
    job_skills = extract_job_skills(job_description)
    must_haves = parse_must_have_requirements(job_description)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name="resume_collection")

    job_embedding = model.encode([job_description])[0].tolist()
    query_result = collection.query(
        query_embeddings=[job_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for idx, doc in enumerate(query_result["documents"][0]):
        meta = query_result["metadatas"][0][idx]
        candidate_name = meta.get("candidate_name", "Unknown Candidate")
        metadata_skills = meta.get("skills", "[]")
        try:
            metadata_skill_list = json.loads(metadata_skills)
        except Exception:
            metadata_skill_list = []

        section = meta.get("section", "general")
        keyword_score = candidate_keyword_overlap(job_skills, doc, metadata_skill_list)
        semantic_score = calculate_similarity(query_result["distances"][0][idx])
        section_factor = SECTION_WEIGHT.get(section.lower(), 0.8)
        raw_score = min(100.0, (semantic_score * 70.0) + (keyword_score * 30.0) * section_factor)
        grouped[candidate_name].append(
            {
                "candidate_name": candidate_name,
                "resume_path": meta.get("resume_path", "unknown"),
                "section": section,
                "match_score": raw_score,
                "excerpt": doc[:250],
                "text": doc,
                "metadata_skills": metadata_skill_list,
                "experience_years": float(meta.get("experience_years", 0) or 0),
            }
        )

    final_matches = []
    for candidate_name, records in grouped.items():
        candidate = evaluate_candidate(records, job_skills, must_haves)
        if candidate["passes_must_have"]:
            final_matches.append(candidate)

    final_matches.sort(key=lambda x: x["match_score"], reverse=True)
    final_matches = final_matches[:10]

    for match in final_matches:
        match.pop("experience_years", None)
        match.pop("sections", None)
        match.pop("passes_must_have", None)

    return {
        "job_description": job_description,
        "top_matches": final_matches,
    }


def main():
    parser = argparse.ArgumentParser(description="Match a job description against indexed resumes.")
    parser.add_argument("--job-description", type=str, help="Full job description to match")
    parser.add_argument("--job-file", type=str, help="Path to a file containing the job description")
    parser.add_argument("--persist-dir", default="vector_store", help="Chroma DB directory")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum number of results to retrieve")
    args = parser.parse_args()

    if args.job_file:
        with open(args.job_file, "r", encoding="utf-8") as f:
            job_description = f.read()
    elif args.job_description:
        job_description = args.job_description
    else:
        raise ValueError("Pass --job-description or --job-file")

    result = match_job_to_resumes(job_description, args.persist_dir, args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
