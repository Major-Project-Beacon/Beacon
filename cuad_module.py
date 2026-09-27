"""
cuad_module.py
--------------
CUAD-based contract compliance branch.

Category prediction now uses your trained classifier (category_classifier/)
instead of nearest-neighbor guessing against category names.

Expected artifact layout:
    models/cuad/
        legalbert_final/          <- QA / clause-extraction model
        clause_embeddings.npy     <- retrieval index (real annotated clauses)
        clause_records.json       <- the clauses those embeddings map to
        category_classifier/      <- NEW: trained category classifier
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
)
from sentence_transformers import SentenceTransformer

DEFAULT_ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "models", "cuad")
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
RETRIEVAL_MODEL_NAME = "BAAI/bge-base-en-v1.5"
TOP_K = 5
TOP_N_CATEGORIES = 2  # how many candidate categories the classifier's top guesses to pool clauses from


class CUADModule:
    SOURCE_NAME = "CUAD"

    def __init__(self, artifact_dir: str = DEFAULT_ARTIFACT_DIR, top_k: int = TOP_K,
                 device: str = None):
        self.artifact_dir = artifact_dir
        self.top_k = top_k
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model_path = os.path.join(artifact_dir, "legalbert_final")
        embeddings_path = os.path.join(artifact_dir, "clause_embeddings.npy")
        records_path = os.path.join(artifact_dir, "clause_records.json")
        classifier_path = os.path.join(artifact_dir, "category_classifier")
        for p in (model_path, embeddings_path, records_path, classifier_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing CUAD artifact: {p}")

        # QA model (used only by extract_from_new_contract)
        self.qa_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(self.device)
        self.qa_model.eval()

        # Retrieval index (real annotated clauses, used to pull the top-k clauses within a category)
        self.retrieval_model = SentenceTransformer(RETRIEVAL_MODEL_NAME)
        self.clause_embeddings = np.load(embeddings_path)
        with open(records_path, "r", encoding="utf-8") as f:
            self.clause_records: List[Dict[str, str]] = json.load(f)

        self.category_to_indices = defaultdict(list)
        for idx, r in enumerate(self.clause_records):
            self.category_to_indices[r["category"]].append(idx)

        # Trained category classifier (replaces old nearest-category-name guessing)
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
            classifier_path
        ).to(self.device)
        self.classifier_model.eval()

        # Compliance comparator
        self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL_NAME
        ).to(self.device)
        self.nli_model.eval()

    # ------------------------------------------------------------------
    def _predict_category(self, meeting_sentence: str, top_n: int = TOP_N_CATEGORIES):
        inputs = self.classifier_tokenizer(
            meeting_sentence, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)
        with torch.no_grad():
            logits = self.classifier_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        top_indices = torch.argsort(probs, descending=True)[:top_n]
        id2label = self.classifier_model.config.id2label
        return [(id2label[int(i)], float(probs[i])) for i in top_indices]

    def _retrieve_top_clauses(self, meeting_sentence: str, categories):
        query_embedding = self.retrieval_model.encode(
            [meeting_sentence], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        indices = []
        for cat in categories:
            indices.extend(self.category_to_indices[cat])
        if not indices:
            return []
        sub_scores = np.dot(self.clause_embeddings[indices], query_embedding)
        ranked = sorted(zip(indices, sub_scores), key=lambda x: x[1], reverse=True)[: self.top_k]
        return [
            {
                "score": float(score),
                "title": self.clause_records[idx]["contract_title"],
                "category": self.clause_records[idx]["category"],
                "clause": self.clause_records[idx]["clause"],
            }
            for idx, score in ranked
        ]

    def _check_compliance(self, meeting_sentence: str, clause_text: str):
        inputs = self.nli_tokenizer(
            clause_text, meeting_sentence,
            return_tensors="pt", truncation=True, max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        label = self.nli_model.config.id2label[int(torch.argmax(probs))].lower()

        if "entail" in label:
            verdict, reason = "COMPLIANT", "The meeting decision matches what this clause requires."
        elif "contradict" in label:
            verdict, reason = "NON-COMPLIANT", "The meeting decision appears to contradict this clause."
        else:
            verdict, reason = "NEEDS REVIEW", "The relationship between the decision and the clause is unclear."

        scores = {self.nli_model.config.id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
        return verdict, reason, scores

    # ------------------------------------------------------------------
    def analyze(self, meeting_sentence: str) -> Dict[str, Any]:
        top_categories = self._predict_category(meeting_sentence)
        category, category_confidence = top_categories[0]
        candidates = self._retrieve_top_clauses(meeting_sentence, [c for c, _ in top_categories])

        checked = []
        for c in candidates:
            verdict, reason, scores = self._check_compliance(meeting_sentence, c["clause"])
            checked.append({**c, "compliance": verdict, "reason": reason, "confidence_scores": scores})

        verdicts = [c["compliance"] for c in checked]
        if "NON-COMPLIANT" in verdicts:
            overall = "NON-COMPLIANT"
        elif "NEEDS REVIEW" in verdicts:
            overall = "NEEDS REVIEW"
        else:
            overall = "COMPLIANT"

        top = next((c for c in checked if c["compliance"] == overall), checked[0] if checked else None)

        return {
            "source": self.SOURCE_NAME,
            "meeting_sentence": meeting_sentence,
            "matched_category": category,
            "category_confidence": category_confidence,
            "candidates_checked": checked,
            "compliance": overall,
            "reason": top["reason"] if top else "No candidate clauses found.",
            "matched_title": top["title"] if top else None,
            "matched_clause": top["clause"] if top else None,
            "retrieval_score": top["score"] if top else 0.0,
        }

    def analyze_batch(self, sentences: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze(s) for s in sentences]

    # ------------------------------------------------------------------
    def extract_from_new_contract(self, category: str, contract_text: str,
                                   max_length: int = 384, doc_stride: int = 128):
        question = (
            f'Highlight the parts (if any) of this contract related to '
            f'"{category}" that should be reviewed by a lawyer.'
        )
        inputs = self.qa_tokenizer(
            question, contract_text,
            truncation="only_second", max_length=max_length, stride=doc_stride,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding="max_length", return_tensors="pt",
        )
        offset_mapping = inputs.pop("offset_mapping")
        inputs.pop("overflow_to_sample_mapping")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.qa_model(**inputs)

        best_score, best_answer = -1e9, ""
        for i in range(inputs["input_ids"].shape[0]):
            start_logits = outputs.start_logits[i].cpu().numpy()
            end_logits = outputs.end_logits[i].cpu().numpy()
            offsets = offset_mapping[i]
            start_idx, end_idx = int(start_logits.argmax()), int(end_logits.argmax())
            if end_idx < start_idx or end_idx - start_idx > 200:
                continue
            score = start_logits[start_idx] + end_logits[end_idx]
            if score > best_score:
                start_char, end_char = offsets[start_idx][0], offsets[end_idx][1]
                if start_char == 0 and end_char == 0:
                    continue
                best_score, best_answer = score, contract_text[start_char:end_char]

        return best_answer.strip(), float(best_score)