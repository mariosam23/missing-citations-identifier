/**
 * Regex helpers for BibTeX key extraction.
 *
 * Deliberately minimal — we only need to detect existing keys in a `.bib`
 * file to prevent duplicate appends.  A full parser is out of scope.
 */

const KEY_PATTERN = /@\w+\s*\{\s*([^,\s]+)\s*,/g;

/**
 * Extract all citation keys from raw `.bib` file content.
 *
 * Matches `@type{key,` with arbitrary whitespace. Does **not** attempt
 * to validate entry bodies.
 */
export function extractKeys(bibContent: string): Set<string> {
  const keys = new Set<string>();
  let match: RegExpExecArray | null;
  while ((match = KEY_PATTERN.exec(bibContent)) !== null) {
    keys.add(match[1]);
  }
  // Reset lastIndex for safety (global regex).
  KEY_PATTERN.lastIndex = 0;
  return keys;
}
