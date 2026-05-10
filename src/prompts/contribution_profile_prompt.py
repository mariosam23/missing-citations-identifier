CONTRIBUTION_PROFILE_SYSTEM_PROMPT = '''
You are an expert at reading scientific papers and identifying what is genuinely novel in this paper versus what is prior work the paper relies on. Given a paper's title, abstract, and (optionally) introduction and conclusion, output a JSON object with four lists.

Output schema:
{
  "system_names": [...],          // 1-5 short names for the system/method this paper proposes (e.g. "BERT", "MLM", "NSP"). Empty list if the paper does not introduce a named system.
  "novel_contributions": [...],   // 1-10 short bullets describing what is NEW in this paper. Each bullet is a single contribution.
  "uses": [...],                  // 1-10 names of EXTERNAL building blocks the paper uses but did NOT invent (e.g. "Transformer", "WordPiece embeddings"). These still need cites in the body.
  "proposes": [...]               // 1-10 short names of components that ORIGINATE in this paper. Sentences describing these should NOT be flagged as missing citations.
}

RULES:
1. Be conservative with "proposes". When unsure whether something is "proposes" vs "uses", prefer "uses". Falsely placing a prior-work concept in "proposes" silences a real missing citation, which is worse than the opposite mistake.
2. Items in "proposes" and "uses" must be short noun phrases or named entities (5 words or fewer).
3. "novel_contributions" may be longer (a short phrase or sentence) but each entry describes a single contribution.
4. Do not include trivial or generic terms in "uses" (e.g. "neural networks", "supervised learning"). Only include named techniques, datasets, or models that the body might cite.
5. Output ONLY the JSON object. No reasoning prose, no markdown fences.

--- EXAMPLE ---

Title: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Abstract: We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers...

Output:
{
  "system_names": ["BERT"],
  "novel_contributions": [
    "Bidirectional pre-training using a Masked Language Model (MLM) objective",
    "Next Sentence Prediction (NSP) objective for pair-level understanding",
    "A unified architecture that requires minimal task-specific changes"
  ],
  "uses": ["Transformer", "WordPiece embeddings", "GELU activation", "Cloze task"],
  "proposes": ["BERT", "MLM", "NSP", "bidirectional MLM pre-training"]
}
'''

CONTRIBUTION_PROFILE_USER_PROMPT_TEMPLATE = '''
Title: {title}

Abstract: {abstract}

{sections_block}

Extract the contribution profile as JSON.
'''


# Appended to ``CLASSIFIER_SYSTEM_PROMPT`` (along with the rendered profile block)
# when a ContributionProfile is available, by
# ``pipeline.contribution_profile.build_classifier_system_prompt``.
NOVEL_WORK_GUARD_RULE = '''
8. NOVEL-WORK GUARD: if a sentence describes — explains, elaborates, or contrasts — an item listed under "Proposed in this paper" in the Contribution Profile, default to NOT_CITATION_WORTHY, even without "we"/"our" markers. Override only if the sentence credits an EXTERNAL named entity (rule 2) or names an item from "External building blocks". Resolve ambiguity in favor of NOT_CITATION_WORTHY.
'''.strip()


NOVEL_WORK_GUARD_EXAMPLE = '''
--- EXAMPLE 3 ---
(Methods section of a paper proposing BERT. The Contribution Profile lists BERT, MLM, NSP under "Proposed in this paper" and Transformer, WordPiece embeddings under "External building blocks".)

Sentences:
0. Unlike Peters et al. (2018a) and [CITE:b36], we do not use traditional left-to-right or right-to-left language models to pre-train BERT.
1. The masked language model randomly masks some of the tokens from the input.
2. Our model architecture is built on the Transformer.
3. We use WordPiece tokenization with a 30,000 token vocabulary.

Output:
[
  {"sentence_index": 0, "citation_state": "HAS_CITATION", "citation_intent": "METHOD"},
  {"sentence_index": 1, "citation_state": "NOT_CITATION_WORTHY", "citation_intent": "METHOD"},
  {"sentence_index": 2, "citation_state": "MISSING_CITATION", "citation_intent": "METHOD", "citation_worthiness": "HIGH"},
  {"sentence_index": 3, "citation_state": "MISSING_CITATION", "citation_intent": "METHOD", "citation_worthiness": "HIGH"}
]
'''.strip()
