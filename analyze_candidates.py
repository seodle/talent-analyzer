import argparse
import logging
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm
import unicodedata

# PDF and DOCX parsing
import pdfplumber
from docx import Document


def setup_logging(verbose: bool, log_file: str) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers = []
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handlers.append(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers)

@dataclass
class CandidateDocs:
    candidate_name: str
    cv_path: Optional[Path]
    letter_path: Optional[Path]
    other_files: List[Path]


def normalize_string(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.lower()


def score_for_type(filename: str, target: str) -> int:
    name = normalize_string(filename)
    # Keywords tuned for French and general cases
    cv_keywords = [
        "cv", "curriculum", "vitae", "curriculum vitae", "résumé", "resume", "c.v"
    ]
    letter_keywords = [
        "lettre", "motivation", "lm", "cover", "letter"
    ]
    keywords = cv_keywords if target == "cv" else letter_keywords
    score = 0
    for kw in keywords:
        if kw in name:
            score += 1
    # Bonus if keyword appears as a separate token-like pattern
    if re.search(r"\b(cv|lm)\b", name):
        score += 1
    return score


def classify_candidate_files(candidate_dir: Path) -> CandidateDocs:
    logging.debug(f"Analyse des fichiers dans: {candidate_dir}")
    files = [p for p in candidate_dir.iterdir() if p.is_file()]
    supported_exts = {".pdf", ".docx"}
    supported_files = [p for p in files if p.suffix.lower() in supported_exts]
    logging.debug(f"Fichiers supportés trouvés ({len(supported_files)}): {[p.name for p in supported_files]}")

    best_cv: Optional[Tuple[int, Path]] = None
    best_letter: Optional[Tuple[int, Path]] = None
    others: List[Path] = []

    for path in supported_files:
        cv_score = score_for_type(path.name, "cv")
        letter_score = score_for_type(path.name, "letter")
        logging.debug(f"Scores {path.name} -> CV:{cv_score} Lettre:{letter_score}")
        if cv_score > 0 and (best_cv is None or cv_score > best_cv[0] or (cv_score == best_cv[0] and path.stat().st_size > best_cv[1].stat().st_size)):
            best_cv = (cv_score, path)
        if letter_score > 0 and (best_letter is None or letter_score > best_letter[0] or (letter_score == best_letter[0] and path.stat().st_size > best_letter[1].stat().st_size)):
            best_letter = (letter_score, path)

    # Collect remaining as others
    chosen = {best_cv[1] for best_cv in [best_cv] if best_cv is not None} | {best_letter[1] for best_letter in [best_letter] if best_letter is not None}
    for p in supported_files:
        if p not in chosen:
            others.append(p)

    if best_cv is None:
        logging.warning(f"CV non détecté pour {candidate_dir.name}")
    else:
        logging.debug(f"CV sélectionné: {best_cv[1].name}")
    if best_letter is None:
        logging.warning(f"Lettre de motivation non détectée pour {candidate_dir.name}")
    else:
        logging.debug(f"Lettre sélectionnée: {best_letter[1].name}")

    return CandidateDocs(
        candidate_name=candidate_dir.name,
        cv_path=best_cv[1] if best_cv else None,
        letter_path=best_letter[1] if best_letter else None,
        other_files=others,
    )


def extract_text_from_pdf(path: Path) -> str:
    texts: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
    return "\n\n".join(texts).strip()


def extract_text_from_docx(path: Path) -> str:
    doc = Document(str(path))
    paras = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paras).strip()


def extract_text_from_file(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        if path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(path)
            logging.debug(f"Texte PDF extrait {path.name} ({len(text)} chars)")
            return text
        if path.suffix.lower() == ".docx":
            text = extract_text_from_docx(path)
            logging.debug(f"Texte DOCX extrait {path.name} ({len(text)} chars)")
            return text
        return ""
    except Exception as exc:
        logging.exception(f"Erreur d'extraction pour {path}: {exc}")
        return f"[Erreur d'extraction: {exc}]"


def truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n[TRONQUÉ]"


def parse_criteria_list(criteria_text: str) -> List[str]:
    items: List[str] = []
    for raw in criteria_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Remove common bullet prefixes
        line = re.sub(r"^(?:[-*•]\s+|\d+\)\s+|\d+\.\s+)", "", line)
        if line:
            items.append(line)
    return items


def read_env() -> Tuple[str, str]:
    load_dotenv()
    product_id = os.getenv("PRODUCT_ID")
    api_key = os.getenv("INFOMANIAK_API_KEY")

    if not product_id or not api_key:
        raise RuntimeError("Variables d'environnement manquantes: PRODUCT_ID et/ou INFOMANIAK_API_KEY dans .env")
    return product_id, api_key


def call_infomaniak_llm(prompt: str, product_id: str, api_key: str, timeout_seconds: int = 120) -> str:
    url = f"https://api.infomaniak.com/1/ai/{product_id}/openai/chat/completions"
    payload = {
        "model": "llama3",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    logging.debug(f"Appel API Infomaniak {url} (prompt {len(prompt)} chars)")
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_seconds)
    if response.status_code != 200:
        logging.error(f"Erreur API {response.status_code}: {response.text[:500]}")
        raise RuntimeError(f"Erreur API {response.status_code}: {response.text[:500]}")
    data = response.json()
    # Common OpenAI-compatible shape
    try:
        content = data["choices"][0]["message"]["content"]
        logging.debug(f"Réponse API reçue ({len(content)} chars)")
        return content
    except Exception:
        # Fallback to alternative shapes
        if isinstance(data, dict) and "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                msg = choice.get("message") or choice.get("delta") or {}
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    logging.debug(f"Réponse API (fallback) reçue ({len(content)} chars)")
                    return content
        return json.dumps(data)


def build_scoring_prompt(criteria_list: List[str], candidate_name: str, cv_text: str, letter_text: str, has_cv: bool, has_letter: bool) -> str:
    cv_text = truncate_text(cv_text.strip(), 12000)
    letter_text = truncate_text(letter_text.strip(), 8000)
    missing = []
    if not has_cv:
        missing.append("CV manquant")
    if not has_letter:
        missing.append("Lettre de motivation manquante")
    missing_note = ", ".join(missing) if missing else ""

    instruction = (
        "Tu es un assistant de recrutement francophone. Évalue ce candidat par rapport aux critères. "
        "Réponds UNIQUEMENT avec un JSON valide, sans texte additionnel. "
        "Le JSON doit contenir les clés: name, cv_relevance, letter_relevance, seniority, overall_score, fit_summary, risks, criteria_checklist. "
        "criteria_checklist est une liste d'objets {criterion, status, source, evidence, confidence}. "
        "status ∈ {met, partial, not_met}. source ∈ {cv, letter, both, none}. confidence ∈ [0,100]. "
        "Pour evidence, cite une courte phrase ou mot-clé trouvé."
    )

    json_schema_hint = {
        "name": candidate_name,
        "cv_relevance": 0,
        "letter_relevance": 0,
        "seniority": 0,
        "overall_score": 0,
        "fit_summary": "",
        "risks": "",
        "criteria_checklist": [
            {"criterion": "", "status": "met", "source": "cv", "evidence": "", "confidence": 0}
        ]
    }

    criteria_block = "\n".join(f"- {c}" for c in criteria_list)
    prompt = (
        f"{instruction}\n\n"
        f"Critères du poste (un par ligne):\n{criteria_block}\n\n"
        f"Informations manquantes: {missing_note if missing_note else 'Aucune'}\n\n"
        f"=== CV ===\n{cv_text}\n\n=== Lettre de motivation ===\n{letter_text}\n\n"
        f"Schéma JSON attendu (exemple de structure, valeurs à recalculer):\n{json.dumps(json_schema_hint, ensure_ascii=False)}"
    )
    return prompt


def try_parse_json(text: str) -> Optional[Dict]:
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Extract first JSON object heuristically
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def safe_int(value: Optional[object]) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, (int, float)):
            return int(round(float(value)))
        if isinstance(value, str):
            num = re.findall(r"-?\d+\.?\d*", value)
            return int(round(float(num[0]))) if num else 0
        return 0
    except Exception:
        return 0


def normalize_status(value: Optional[str]) -> str:
    if not value:
        return "not_met"
    v = str(value).strip().lower()
    if v in {"met", "ok", "oui", "yes", "true", "satisfait"}:
        return "met"
    if v in {"partial", "partiel", "partially_met", "partiellement"}:
        return "partial"
    return "not_met"


def normalize_source(value: Optional[str]) -> str:
    if not value:
        return "none"
    v = str(value).strip().lower()
    if v in {"cv", "resume"}:
        return "cv"
    if v in {"letter", "lettre", "cover"}:
        return "letter"
    if v in {"both", "cv+letter", "cv_letter"}:
        return "both"
    return "none"


def evaluate_candidate(criteria_list: List[str], docs: CandidateDocs, product_id: str, api_key: str) -> Dict:
    cv_text = extract_text_from_file(docs.cv_path)
    letter_text = extract_text_from_file(docs.letter_path)
    has_cv = docs.cv_path is not None and bool(cv_text.strip())
    has_letter = docs.letter_path is not None and bool(letter_text.strip())
    logging.info(f"Évaluation {docs.candidate_name} (CV:{has_cv} Lettre:{has_letter})")

    prompt = build_scoring_prompt(criteria_list, docs.candidate_name, cv_text, letter_text, has_cv, has_letter)

    try:
        response_text = call_infomaniak_llm(prompt, product_id, api_key)
        parsed = try_parse_json(response_text)
    except Exception as exc:
        parsed = None
        response_text = f"[Erreur API: {exc}]"
        logging.exception(f"Échec appel LLM pour {docs.candidate_name}: {exc}")

    result: Dict[str, object] = {
        "candidate_name": docs.candidate_name,
        "has_cv": has_cv,
        "has_letter": has_letter,
        "cv_path": str(docs.cv_path) if docs.cv_path else "",
        "letter_path": str(docs.letter_path) if docs.letter_path else "",
        "raw_response": response_text,
    }

    if parsed is None:
        # Fallback minimal scoring when LLM fails
        result.update({
            "cv_relevance": 0,
            "letter_relevance": 0,
            "seniority": 0,
            "overall_score": 0,
            "fit_summary": "Échec de l'analyse LLM",
            "risks": "Réponse non exploitable",
        })
        return result

    cv_rel = safe_int(parsed.get("cv_relevance"))
    lt_rel = safe_int(parsed.get("letter_relevance"))
    seniority = safe_int(parsed.get("seniority"))
    overall = safe_int(parsed.get("overall_score"))

    # Criteria checklist
    raw_checklist = parsed.get("criteria_checklist") if isinstance(parsed, dict) else None
    checklist: List[Dict[str, object]] = []
    if isinstance(raw_checklist, list):
        for item in raw_checklist:
            if not isinstance(item, dict):
                continue
            criterion = str(item.get("criterion", "")).strip()
            if not criterion:
                continue
            status = normalize_status(item.get("status"))
            source = normalize_source(item.get("source"))
            evidence = str(item.get("evidence", "")).strip()
            confidence = safe_int(item.get("confidence"))
            checklist.append({
                "criterion": criterion,
                "status": status,
                "source": source,
                "evidence": evidence,
                "confidence": confidence,
            })

    # Ensure every criterion from criteria_list appears once
    found_map = {str(item.get("criterion", "")).strip(): item for item in checklist}
    completed_checklist: List[Dict[str, object]] = []
    for crit in criteria_list:
        if crit in found_map and found_map[crit]:
            completed_checklist.append(found_map[crit])
        else:
            completed_checklist.append({
                "criterion": crit,
                "status": "not_met",
                "source": "none",
                "evidence": "",
                "confidence": 0,
            })

    result.update({
        "cv_relevance": cv_rel,
        "letter_relevance": lt_rel,
        "seniority": seniority,
        "overall_score": overall,
        "fit_summary": str(parsed.get("fit_summary", "")).strip(),
        "risks": str(parsed.get("risks", "")).strip(),
        "criteria_checklist": completed_checklist,
    })
    return result


def find_candidate_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def save_csv(rows: List[Dict], out_path: Path) -> None:
    fieldnames = [
        "candidate_name",
        "overall_score",
        "cv_relevance",
        "letter_relevance",
        "seniority",
        "has_cv",
        "has_letter",
        "cv_path",
        "letter_path",
        "fit_summary",
        "risks",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_markdown(rows: List[Dict], out_path: Path, top_n: int) -> None:
    lines: List[str] = []
    lines.append("# Classement des candidats\n")
    for idx, row in enumerate(rows[:top_n], start=1):
        lines.append(f"## {idx}. {row['candidate_name']} — Score global: {row['overall_score']}")
        lines.append("")
        lines.append(f"- CV: {row['cv_relevance']} | Lettre: {row['letter_relevance']} | Séniorité: {row['seniority']}")
        lines.append(f"- CV trouvé: {row['has_cv']} | Lettre trouvée: {row['has_letter']}")
        # Criteria checklist before synthesis
        checklist = row.get("criteria_checklist") or []
        if checklist:
            lines.append("")
            lines.append("**Critères**:")
            lines.append("")
            for item in checklist:
                try:
                    crit = item.get("criterion", "")
                    status = item.get("status", "")
                    source = item.get("source", "")
                    evidence = item.get("evidence", "")
                    conf = item.get("confidence", 0)
                    # Display mapping
                    src_label = {
                        "cv": "CV",
                        "letter": "Lettre",
                        "both": "CV+Lettre",
                        "none": "Aucune"
                    }.get(str(source).lower(), str(source))
                    status_label = {
                        "met": "OK",
                        "partial": "Partiel",
                        "not_met": "Non"
                    }.get(str(status).lower(), str(status))
                    ev = f" — Extrait: {evidence}" if evidence else ""
                    lines.append(f"- {crit} → {status_label} (Source: {src_label}, Confiance: {conf}){ev}")
                except Exception:
                    continue

        if row.get("fit_summary"):
            lines.append("")
            lines.append("**Synthèse**:")
            lines.append("")
            lines.append(row["fit_summary"])
        if row.get("risks"):
            lines.append("")
            lines.append("**Risques**:")
            lines.append("")
            lines.append(row["risks"])
        lines.append("")

    missing: List[str] = []
    for row in rows:
        notes = []
        if not row["has_cv"]:
            notes.append("CV manquant")
        if not row["has_letter"]:
            notes.append("Lettre manquante")
        if notes:
            missing.append(f"- {row['candidate_name']}: {', '.join(notes)}")
    if missing:
        lines.append("---\n")
        lines.append("## Dossiers incomplets\n")
        lines.extend(missing)
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse de candidats avec Llama 3 (Infomaniak)")
    parser.add_argument("--criteria", type=str, default="", help="(Obsolète) Critères inline — le script lit désormais criteria.txt")
    parser.add_argument("--criteria-file", type=str, default="criteria.txt", help="Fichier texte avec les critères du poste (par défaut: criteria.txt)")
    parser.add_argument("--candidates-dir", type=str, default="candidats", help="Dossier contenant les dossiers des candidats")
    parser.add_argument("--output-dir", type=str, default=".", help="Dossier de sortie pour les rapports")
    parser.add_argument("--top", type=int, default=10, help="Nombre de candidats à afficher dans le rapport Markdown")
    parser.add_argument("--limit", type=int, default=0, help="Limiter l'analyse aux N premiers dossiers (0 = tous)")
    parser.add_argument("--candidate", type=str, default="", help="Analyser un candidat précis (format dossier: 'NOM Prénom')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Activer les logs détaillés (DEBUG)")
    parser.add_argument("--log-file", type=str, default="", help="Écrire les logs dans ce fichier")
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)

    # Critères toujours lus à partir d'un fichier texte (par défaut criteria.txt)
    criteria_path = Path(args.criteria_file)
    if not criteria_path.exists():
        logging.error(f"Fichier de critères introuvable: {criteria_path}")
        print(f"Fichier de critères introuvable: {criteria_path}. Fournissez --criteria-file ou créez criteria.txt", file=sys.stderr)
        sys.exit(1)
    criteria_text = criteria_path.read_text(encoding="utf-8").strip()
    if not criteria_text:
        logging.error(f"Le fichier de critères est vide: {criteria_path}")
        print(f"Le fichier de critères est vide: {criteria_path}", file=sys.stderr)
        sys.exit(1)

    product_id, api_key = read_env()

    candidates_root = Path(args.candidates_dir)
    if not candidates_root.exists():
        logging.error(f"Dossier candidats introuvable: {candidates_root}")
        print(f"Dossier candidats introuvable: {candidates_root}", file=sys.stderr)
        sys.exit(1)

    candidate_dirs = find_candidate_dirs(candidates_root)
    logging.info(f"Dossiers candidats trouvés: {len(candidate_dirs)}")
    if args.candidate:
        wanted = args.candidate.strip().lower()
        # Normalize accents and spaces for robust matching
        def nfkd(s: str) -> str:
            import unicodedata as _ud
            s2 = _ud.normalize("NFKD", s)
            s2 = "".join(ch for ch in s2 if not _ud.combining(ch))
            return re.sub(r"\s+", " ", s2).strip().lower()

        wanted_n = nfkd(wanted)
        filtered = [p for p in candidate_dirs if nfkd(p.name) == wanted_n]
        if not filtered:
            logging.error(f"Candidat non trouvé: {args.candidate}")
            print(f"Candidat non trouvé: {args.candidate}", file=sys.stderr)
            sys.exit(1)
        candidate_dirs = filtered
        logging.info(f"Filtre --candidate appliqué: {candidate_dirs[0].name}")
    if not args.candidate and args.limit and args.limit > 0:
        candidate_dirs = candidate_dirs[: args.limit]
        logging.info(f"Limitation active: analyse des {len(candidate_dirs)} premiers dossiers")
        print(f"Limitation active: analyse des {len(candidate_dirs)} premiers dossiers")
    if not candidate_dirs:
        logging.warning("Aucun dossier candidat trouvé.")
        print("Aucun dossier candidat trouvé.", file=sys.stderr)
        sys.exit(1)

    results: List[Dict] = []
    crit_list = parse_criteria_list(criteria_text)
    for cand_dir in tqdm(candidate_dirs, desc="Analyse des candidats"):
        logging.debug(f"Traitement dossier: {cand_dir.name}")
        docs = classify_candidate_files(cand_dir)
        result = evaluate_candidate(crit_list, docs, product_id, api_key)
        results.append(result)

    # Sort by overall_score desc
    results.sort(key=lambda r: r.get("overall_score", 0), reverse=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "report_candidates.csv"
    md_path = output_dir / "report_candidates.md"

    save_csv(results, csv_path)
    save_markdown(results, md_path, args.top)

    print(f"Rapports générés:\n- {csv_path}\n- {md_path}")
    if args.log_file:
        print(f"Logs détaillés: {args.log_file}")

if __name__ == "__main__":
    main()
