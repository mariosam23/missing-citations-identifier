export interface PaperLike {
  citation_key: string;
  authors: string[];
  year: number | null;
}

const LATEX_LANGUAGES = new Set(["latex", "tex", "bibtex"]);
const MARKDOWN_LANGUAGES = new Set([
  "markdown",
  "md",
  "quarto",
  "qmd",
  "rmd",
  "mdx",
]);

/**
 * Render a citation marker appropriate to the host document.
 *
 * - LaTeX/TeX → `\cite{key}`
 * - Markdown / Quarto / R Markdown → `[@key]` (Pandoc-style)
 * - Anything else → `(Surname et al., Year)` plain-text fallback
 *
 * `selection` is reserved for callers that want to wrap rather than replace
 * the highlight; the MVP simply replaces it.
 */
export function formatCitation(
  paper: PaperLike,
  languageId: string | undefined,
): string {
  const lang = (languageId ?? "").toLowerCase();

  if (LATEX_LANGUAGES.has(lang)) {
    return `\\cite{${paper.citation_key}}`;
  }
  if (MARKDOWN_LANGUAGES.has(lang)) {
    return `[@${paper.citation_key}]`;
  }
  return formatPlainText(paper);
}

function formatPlainText(paper: PaperLike): string {
  const year = paper.year ?? "n.d.";
  const surname = firstAuthorSurname(paper.authors);
  if (!surname) {
    return `(${paper.citation_key}, ${year})`;
  }
  const multipleAuthors = paper.authors.length > 1;
  const tail = multipleAuthors ? " et al." : "";
  return `(${surname}${tail}, ${year})`;
}

function firstAuthorSurname(authors: string[]): string | null {
  if (authors.length === 0) {
    return null;
  }
  const raw = authors[0].trim();
  if (!raw) {
    return null;
  }
  // OpenAlex shape sometimes "Last, First"; GROBID/SBSS may be "First Last".
  if (raw.includes(",")) {
    return raw.split(",", 1)[0].trim();
  }
  const tokens = raw.split(/\s+/);
  return tokens[tokens.length - 1];
}
