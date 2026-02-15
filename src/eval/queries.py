"""20-query evaluation set for the Exercise & Nutrition for Longevity RAG.

Queries are tailored to the actual 18-paper corpus so that each direct
query targets a known paper and each edge case tests a specific guardrail.

Categories:
  - 10 direct / factual queries
  - 5 synthesis / cross-paper queries
  - 5 edge-case / adversarial queries
"""

EVAL_QUERIES: list[dict] = [
    # ── Direct / Factual (10) ─────────────────────────────────────────────────
    # Each targets 1-2 specific papers in the corpus.
    {
        "id": "D01",
        "type": "direct",
        "query": "What biomarkers did Brandhorst et al. use to measure biological age changes from the fasting-mimicking diet?",
        "target_papers": ["brandhorst_2024"],
    },
    {
        "id": "D02",
        "type": "direct",
        "query": "How does the AMPK/SIRT1/PGC-1α signaling axis respond to exercise?",
        "target_papers": ["mol_bio_rep_2025"],
    },
    {
        "id": "D03",
        "type": "direct",
        "query": "What role does autophagy play in exercise-induced mitochondrial biogenesis?",
        "target_papers": ["ju_2016"],
    },
    {
        "id": "D04",
        "type": "direct",
        "query": "According to Barry et al., does cardiorespiratory fitness or body fatness have a stronger effect on all-cause mortality?",
        "target_papers": ["barry_2014"],
    },
    {
        "id": "D05",
        "type": "direct",
        "query": "What are the key criteria for evaluating aging biomarkers according to the Cell 2023 review?",
        "target_papers": ["cell_2023"],
    },
    {
        "id": "D06",
        "type": "direct",
        "query": "How does long-term aerobic exercise preserve muscle mass during aging?",
        "target_papers": ["s2468867319_2019"],
    },
    {
        "id": "D07",
        "type": "direct",
        "query": "What lifestyle factors from Blue Zones are associated with longevity?",
        "target_papers": ["nutrients_2025"],
    },
    {
        "id": "D08",
        "type": "direct",
        "query": "What epigenetic biomarkers change in response to dietary and exercise interventions?",
        "target_papers": ["nutrients_2023a"],
    },
    {
        "id": "D09",
        "type": "direct",
        "query": "What minimum weekly physical activity threshold is associated with reduced mortality risk?",
        "target_papers": ["stamatakis_2025", "s2589537025_2025"],
    },
    {
        "id": "D10",
        "type": "direct",
        "query": "What metabolic biomarkers improved with concurrent aerobic and resistance training in type 2 diabetes patients?",
        "target_papers": ["diabetol_metab_2025"],
    },

    # ── Synthesis / Cross-paper (5) ───────────────────────────────────────────
    # Each requires pulling evidence from 2+ papers and comparing.
    {
        "id": "S01",
        "type": "synthesis",
        "query": "Compare the molecular aging biomarker frameworks in Moqri et al. (Cell 2023) versus Furrer & Handschin (Physiol Rev 2025) — where do they agree and disagree?",
        "target_papers": ["cell_2023", "furrer_handschin_2025"],
    },
    {
        "id": "S02",
        "type": "synthesis",
        "query": "How do caloric restriction, intermittent fasting, and fasting-mimicking diets compare in their mechanisms and evidence for longevity benefits?",
        "target_papers": ["f1000_2016", "nutrients_2024", "brandhorst_2024"],
    },
    {
        "id": "S03",
        "type": "synthesis",
        "query": "Synthesize how exercise affects mitochondrial function across the corpus — what molecular pathways are shared across studies?",
        "target_papers": ["biology_2019", "ju_2016", "mol_bio_rep_2025", "cells_2022"],
    },
    {
        "id": "S04",
        "type": "synthesis",
        "query": "What do the Stamatakis, Koemel, and NHANES studies collectively suggest about the optimal combination of sleep, physical activity, and nutrition for longevity?",
        "target_papers": ["stamatakis_2025", "s2589537025_2025", "kwaa_2020"],
    },
    {
        "id": "S05",
        "type": "synthesis",
        "query": "Across the corpus, what methodological limitations are most commonly acknowledged in human longevity research?",
        "target_papers": [],  # broad — tests retrieval breadth
    },

    # ── Edge-case / Adversarial (5) ───────────────────────────────────────────
    # Each tests a specific trust guardrail.
    {
        "id": "E01",
        "type": "edge",
        "query": "What does the corpus say about the effect of smoking cessation on longevity?",
        "expected_behavior": "Should say evidence is limited or absent — smoking is barely covered in the corpus.",
        "expected_no_evidence": False,
    },
    {
        "id": "E02",
        "type": "edge",
        "query": "Summarize the findings from Zhang et al. 2021 on ketogenic diets and aging.",
        "expected_behavior": "Should refuse — Zhang 2021 does not exist in the corpus.",
        "expected_no_evidence": True,
    },
    {
        "id": "E03",
        "type": "edge",
        "query": "According to Barry et al. 2014, what did the Methods section describe about how studies were selected for the meta-analysis?",
        "expected_behavior": "Should retrieve from barry_2014 methods section specifically; tests section-aware chunking.",
        "expected_no_evidence": False,
    },
    {
        "id": "E04",
        "type": "edge",
        "query": "Is there evidence that caloric restriction is harmful to longevity?",
        "expected_behavior": "Should present evidence fairly, noting CR has both benefits and risks/limitations rather than agreeing with the leading premise.",
        "expected_no_evidence": False,
    },
    {
        "id": "E05",
        "type": "edge",
        "query": "What pharmaceutical drugs does the corpus recommend for extending lifespan?",
        "expected_behavior": "Should note the corpus focuses on lifestyle interventions (exercise/diet), not drug recommendations.",
        "expected_no_evidence": False,
    },
]
