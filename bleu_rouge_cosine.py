import argparse

import numpy as np
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REF_COL = "Teacher Question"
NO_PE_COL = "AI questions generated 1"
WITH_PE_COL = "AI Questions Generated 2"
SUBJECT_COL = "School Subject"


def calculate_metrics(subset):
    refs = subset[REF_COL].astype(str).tolist()
    cands_no = subset[NO_PE_COL].astype(str).tolist()
    cands_pe = subset[WITH_PE_COL].astype(str).tolist()

    # BLEU: SacreBLEU corpus score
    bleu_no = sacrebleu.corpus_bleu(cands_no, [refs]).score
    bleu_pe = sacrebleu.corpus_bleu(cands_pe, [refs]).score

    # ROUGE-L: mean pairwise F1 score, with stemming
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_no = (
        np.mean(
            [
                scorer.score(reference, candidate)["rougeL"].fmeasure
                for reference, candidate in zip(refs, cands_no)
            ]
        )
        * 100
    )
    rouge_pe = (
        np.mean(
            [
                scorer.score(reference, candidate)["rougeL"].fmeasure
                for reference, candidate in zip(refs, cands_pe)
            ]
        )
        * 100
    )

    # TF-IDF cosine similarity: mean pairwise score
    def get_avg_cosine(candidates, references):
        similarities = []

        for candidate, reference in zip(candidates, references):
            vectors = TfidfVectorizer().fit_transform([candidate, reference])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            similarities.append(similarity)

        return np.mean(similarities)

    cosine_no = get_avg_cosine(cands_no, refs)
    cosine_pe = get_avg_cosine(cands_pe, refs)

    return {
        "BLEU_No": bleu_no,
        "BLEU_PE": bleu_pe,
        "ROUGE-L_No": rouge_no,
        "ROUGE-L_PE": rouge_pe,
        "TF-IDF Cosine_No": cosine_no,
        "TF-IDF Cosine_PE": cosine_pe,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare unprompted vs prompted AI questions using BLEU, ROUGE-L, and TF-IDF cosine similarity."
    )
    parser.add_argument(
        "--input",
        default="AP Research Data Gathering Collection Form - Sheet1 (1).csv",
        help="Path to the input CSV.",
    )
    parser.add_argument(
        "--output",
        default="final_comparative_metrics_report.csv",
        help="Path for the output CSV.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required_columns = [SUBJECT_COL, REF_COL, NO_PE_COL, WITH_PE_COL]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Do not silently convert missing values to the string "nan".
    missing_counts = df[[REF_COL, NO_PE_COL, WITH_PE_COL]].isna().sum()
    if missing_counts.any():
        raise ValueError(
            "Missing values found in evaluation columns:\n"
            + missing_counts.to_string()
        )

    summary_list = []

    for subject in df[SUBJECT_COL].dropna().unique():
        subject_df = df[df[SUBJECT_COL] == subject]
        metrics = calculate_metrics(subject_df)

        metric_map = {
            "BLEU": ("BLEU_No", "BLEU_PE"),
            "ROUGE-L": ("ROUGE-L_No", "ROUGE-L_PE"),
            "TF-IDF Cosine": ("TF-IDF Cosine_No", "TF-IDF Cosine_PE"),
        }

        for metric_name, (no_key, pe_key) in metric_map.items():
            no_value = metrics[no_key]
            pe_value = metrics[pe_key]

            percent_change = (
                ((pe_value - no_value) / no_value) * 100
                if no_value != 0
                else np.nan
            )

            summary_list.append(
                {
                    "Subject": subject,
                    "Metric": metric_name,
                    "Without Prompt Engineering": no_value,
                    "With Prompt Engineering": pe_value,
                    "% Change": percent_change,
                }
            )

    summary_df = pd.DataFrame(summary_list)
    print(summary_df.to_string(index=False))

    output_path = Path(args.output)
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
