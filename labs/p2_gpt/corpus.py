"""Corpora for Project 2, with licence provenance recorded per source.

WHY THIS FILE EXISTS SEPARATELY FROM data.py
--------------------------------------------
Dataset licensing is the part of a portfolio most likely to be wrong and least likely to be checked.
Putting corpus acquisition in its own module means the licence of every byte we train on is stated in
one auditable place, with the exact retrieval parameters needed to reconstruct it.

--------------------------------------------------------------------------------------------
CORPUS: "design"  (the default)
--------------------------------------------------------------------------------------------
Graphic design, assembled from two sources with different licences, kept separable so either can be
used alone.

**Source A — Project Gutenberg, public domain.**
Books on typography, printing and page design. Their copyright has expired, so there is no
share-alike obligation and no ambiguity about the licence status of a model trained on them.

Why these books are genuinely on-topic rather than merely old: they are about type anatomy, page
proportion, legibility, composition, and the discipline of setting a page well. Those are design
fundamentals, not printing trivia, and the vocabulary transfers.

Their limitation, stated because an audience will notice: they are ~1900s craft prose. They will not
teach the model "grid system", "Bauhaus", "kerning" in its modern sense, "brand identity", or
anything about screens. That is what Source B is for.

**Source B — Wikipedia, CC BY-SA 4.0.**
A curated list of graphic-design articles: typography, colour theory, layout, movements, and the
designers who defined them. This supplies modern vocabulary the public-domain books cannot.

Attribution is recorded per article, including the **exact revision id**, so the corpus is
reconstructible byte-for-byte and the attribution points at the specific text used rather than at a
page that has since changed. `attribution.json` in the data directory is the licence record.

One honest caveat about share-alike: CC BY-SA requires derivative works to be licensed alike. Whether
a trained model's weights constitute a derivative work of its training text is legally unsettled and
this project does not pretend to resolve it. Mitigation: the public-domain source can be used alone
(`--corpus design_pd`) if that matters, the Wikipedia contribution is separable and documented, and
attribution is published regardless.

--------------------------------------------------------------------------------------------
CORPUS: "shakespeare"
--------------------------------------------------------------------------------------------
TinyShakespeare, kept as an alternative. Public domain text, MIT-licensed compilation. Useful as a
control: the same architecture on a different corpus, which makes it possible to say what is a
property of the model versus a property of the data.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CORPORA", "load_corpus", "CorpusSpec"]

USER_AGENT = (
    "ml-from-first-principles/0.1 (https://github.com/zexoraai/ml-from-first-principles; "
    "educational research corpus) python-urllib"
)

# --------------------------------------------------------------------------------------------
# Project Gutenberg: verified public-domain works on typography, printing and page design.
# IDs confirmed against gutenberg.org before inclusion -- none of these are guesses.
# --------------------------------------------------------------------------------------------
GUTENBERG_BOOKS: list[tuple[int, str, str]] = [
    (51034, "The Invention of Printing", "Theodore Low De Vinne"),
    (77910, "In the Day's Work (essays incl. 'Style in the Use of Type')", "Daniel Berkeley Updike"),
    (41289, "The Magazine Style-code", "Leigh H. Irvine"),
    (23754, "Title Pages as Seen by a Printer", "Theodore Low De Vinne"),
]

# --------------------------------------------------------------------------------------------
# Wikipedia: curated graphic-design article list.
#
# Chosen to cover the vocabulary a designer actually uses: type, colour, layout, movements,
# practitioners, and production. Deliberately excludes software product pages, which are mostly
# release-history tables and would teach the model nothing but version numbers.
# --------------------------------------------------------------------------------------------
WIKIPEDIA_ARTICLES: list[str] = [
    # discipline and process
    "Graphic design", "Visual communication", "Design", "Design thinking", "Art direction",
    "Corporate identity", "Brand", "Logo", "Signage", "Information design",
    "Data and information visualization", "Infographic", "User experience design",
    "User interface design", "Interaction design", "Web design", "Editorial design",
    "Book design", "Poster", "Packaging and labeling", "Advertising",
    # typography
    "Typography", "Typeface", "Font", "Serif", "Sans-serif", "Slab serif", "Script typeface",
    "Blackletter", "Monospaced font", "Display typeface", "Type design", "Type foundry",
    "Kerning", "Letter-spacing", "Leading", "Baseline (typography)", "X-height", "Cap height",
    "Ascender (typography)", "Descender", "Ligature (writing)", "Counter (typography)",
    "Point (typography)", "Em (typography)", "En (typography)", "Pica (typography)",
    "Typographic alignment", "Justification (typography)", "Widows and orphans",
    "Hyphenation algorithm", "Small caps", "Italic type", "Oblique type", "Font hinting",
    "Web typography", "Variable font", "OpenType", "TrueType", "PostScript",
    "Legibility", "Readability", "Hierarchy (typography)",
    # named typefaces
    "Helvetica", "Univers", "Akzidenz-Grotesk", "Futura (typeface)", "Bodoni", "Garamond",
    "Caslon", "Baskerville", "Times New Roman", "Gill Sans", "Frutiger (typeface)",
    "Optima", "Palatino", "Century Schoolbook", "Franklin Gothic", "Didot (typeface)",
    "Comic Sans", "Arial", "Georgia (typeface)", "Verdana", "Johnston (typeface)",
    # colour
    "Color theory", "Color", "Color wheel", "Complementary colors", "Analogous colors",
    "Color scheme", "Color model", "RGB color model", "CMYK color model", "HSL and HSV",
    "CIELAB color space", "Color space", "Color management", "Pantone", "Spot color",
    "Color depth", "Color blindness", "Contrast (vision)", "Saturation (color)",
    "Lightness", "Hue", "Additive color", "Subtractive color", "Color temperature",
    # layout and composition
    "Page layout", "Grid (graphic design)", "Golden ratio", "Rule of thirds", "Whitespace",
    "Composition (visual arts)", "Visual hierarchy", "Gestalt psychology", "Symmetry",
    "Balance (design)", "Rhythm", "Proportion (architecture)", "Canons of page construction",
    "Margin (typography)", "Column (typography)", "Modular scale", "Responsive web design",
    # movements and history
    "Bauhaus", "International Typographic Style", "Swiss Style (design)", "De Stijl",
    "Constructivism (art)", "Art Nouveau", "Art Deco", "Modernism", "Postmodernism",
    "Arts and Crafts movement", "Futurism", "Dada", "Psychedelic art",
    "Punk visual art", "Minimalism", "Flat design", "Material Design", "Skeuomorph",
    "History of graphic design", "History of printing", "History of Western typography",
    # practitioners
    "Paul Rand", "Massimo Vignelli", "Josef Müller-Brockmann", "Jan Tschichold",
    "Herb Lubalin", "Saul Bass", "Milton Glaser", "Emil Ruder", "Armin Hofmann",
    "Adrian Frutiger", "Max Miedinger", "Eric Gill", "William Morris", "El Lissitzky",
    "Alexander Rodchenko", "László Moholy-Nagy", "Wolfgang Weingart", "David Carson",
    "Neville Brody", "Paula Scher", "Stefan Sagmeister", "Erik Spiekermann",
    "Susan Kare", "Dieter Rams", "Bruno Munari", "Alvin Lustig", "Cipe Pineles",
    # production and technical
    "Printing", "Offset printing", "Letterpress printing", "Screen printing", "Lithography",
    "Halftone", "Dither", "Bleed (printing)", "Imposition", "Prepress", "Dots per inch",
    "Image resolution", "Raster graphics", "Vector graphics", "Scalable Vector Graphics",
    "Bézier curve", "Anti-aliasing", "Alpha compositing", "Blend modes", "Rasterisation",
    "Portable Document Format", "Desktop publishing", "Paper size", "ISO 216",
    # accessibility and standards
    "Web Content Accessibility Guidelines", "Accessibility", "Universal design",
    "Icon (computing)", "Pictogram", "ISO 7001", "Wayfinding",
]


@dataclass
class CorpusSpec:
    """Everything needed to reconstruct and attribute one corpus.

    `segments` keeps each source's text separate rather than concatenating immediately, because the
    train/validation split must be taken **per source**.

    Why that matters: a single contiguous cut at the end of a concatenated corpus would put one entire
    register into validation. With Wikipedia first and the public-domain books second, validation would
    be nothing but 1900s printing prose while training was mostly modern encyclopedia text — so the
    validation loss would measure out-of-distribution generalisation, not held-out performance, and it
    would look far worse than the model deserved. Splitting each source 90/10 and then concatenating
    keeps both registers represented on both sides in proportion, while staying contiguous *within* a
    source so that overlapping windows cannot straddle the boundary.
    """
    name: str
    description: str
    sources: list[dict] = field(default_factory=list)
    segments: list[tuple[str, str]] = field(default_factory=list)   # (source label, text)

    @property
    def text(self) -> str:
        return "\n\n".join(t for _, t in self.segments)

    def manifest(self) -> dict:
        return {
            "corpus": self.name,
            "description": self.description,
            "total_chars": len(self.text),
            "segments": [{"label": label, "chars": len(t)} for label, t in self.segments],
            "sources": self.sources,
        }

    def split(self, val_fraction: float) -> tuple[str, str]:
        """Contiguous per-segment split. Returns `(train_text, val_text)`."""
        train_parts, val_parts = [], []
        for _, text in self.segments:
            cut = int(len(text) * (1.0 - val_fraction))
            train_parts.append(text[:cut])
            val_parts.append(text[cut:])
        return "\n\n".join(train_parts), "\n\n".join(val_parts)


def _fetch(url: str, *, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return r.read()


# --------------------------------------------------------------------------------------------
# Project Gutenberg
# --------------------------------------------------------------------------------------------

_PG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_PG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)


def strip_gutenberg_boilerplate(raw: str) -> str:
    """Remove the Project Gutenberg header and licence footer.

    Two reasons this matters beyond tidiness. The boilerplate is identical across every book, so
    leaving it in would have BPE spend merges on legal phrasing repeated N times and teach the model
    to generate licence text. And the PG licence terms attach to the *distribution wrapper*, not to
    the public-domain work — stripping it is what keeps the remaining text unambiguously public
    domain.
    """
    start = _PG_START.search(raw)
    if start:
        raw = raw[start.end():]
    end = _PG_END.search(raw)
    if end:
        raw = raw[:end.start()]
    # Transcriber's notes and production credits sit at the very top of many PG texts.
    raw = re.sub(r"^\s*(Transcriber'?s? Note.*?)\n\n", "", raw, flags=re.I | re.S)
    return raw.strip()


def normalise_whitespace(text: str) -> str:
    """Collapse hard-wrapped lines into paragraphs; keep paragraph breaks.

    Public-domain texts are hard-wrapped at ~70 columns, an artefact of the transcription rather than
    of the writing. Left alone, the model learns to emit a newline every ~70 characters, which looks
    absurd next to modern prose. Joining within a paragraph while preserving blank lines keeps the
    structure that carries meaning and discards the structure that does not.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", text)
    out = []
    for p in paragraphs:
        joined = re.sub(r"\s*\n\s*", " ", p).strip()
        joined = re.sub(r"[ \t]{2,}", " ", joined)
        if joined:
            out.append(joined)
    return "\n\n".join(out)


def load_gutenberg(cache_dir: Path, *, verbose: bool = True) -> CorpusSpec:
    spec = CorpusSpec(
        name="design_pd",
        description="Public-domain books on typography, printing and page design (Project Gutenberg)",
    )
    parts: list[str] = []
    for book_id, title, author in GUTENBERG_BOOKS:
        cache = cache_dir / f"pg_{book_id}.txt"
        if cache.exists():
            raw = cache.read_text(encoding="utf-8", errors="replace")
        else:
            url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
            if verbose:
                print(f"  fetching PG {book_id}: {title}")
            try:
                raw = _fetch(url).decode("utf-8", errors="replace")
            except (urllib.error.URLError, TimeoutError) as exc:
                if verbose:
                    print(f"    SKIPPED ({exc.__class__.__name__}: {exc})")
                continue
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(raw, encoding="utf-8")
            time.sleep(1.0)   # be a polite client

        body = normalise_whitespace(strip_gutenberg_boilerplate(raw))
        if len(body) < 2000:
            if verbose:
                print(f"    SKIPPED: only {len(body)} chars after stripping — likely a fetch problem")
            continue
        parts.append(body)
        spec.sources.append({
            "kind": "project_gutenberg",
            "id": book_id,
            "title": title,
            "author": author,
            "url": f"https://www.gutenberg.org/ebooks/{book_id}",
            "licence": "public domain (copyright expired); PG boilerplate stripped",
            "chars": len(body),
        })
        if verbose:
            print(f"    {len(body):>9,} chars  {title}")

    if parts:
        spec.segments = [("project_gutenberg", "\n\n".join(parts))]
    return spec


# --------------------------------------------------------------------------------------------
# Wikipedia
# --------------------------------------------------------------------------------------------

WIKI_API = "https://en.wikipedia.org/w/api.php"


def load_wikipedia(
    cache_dir: Path, titles: list[str] | None = None, *, verbose: bool = True
) -> CorpusSpec:
    """Fetch plain-text extracts plus the revision id of each article.

    Recording the revision id is what makes the corpus reproducible *and* the attribution honest: the
    credit points at the exact text used, not at a page that may since have been rewritten.

    ONE ARTICLE PER REQUEST, and that is not laziness
    -------------------------------------------------
    The first version of this batched 20 titles per call with `exlimit=20`. Every article came back
    without an `extract` field and the code reported them all as "missing". They were not missing:
    MediaWiki's TextExtracts extension **caps full-article extracts at one per request** — `exlimit`
    above 1 only applies when `exintro` restricts output to the lead section. Batching therefore
    returned one usable extract and 19 silently empty pages.

    Worth recording because of how the bug presented: not an error, not an exception, just a log full
    of plausible-looking "missing: Typography" lines. Had the fallback been "skip quietly and carry
    on", the corpus would have been one article and the model would have been trained on it.
    """
    titles = titles or WIKIPEDIA_ARTICLES
    spec = CorpusSpec(
        name="design_wiki",
        description="Curated graphic-design articles from English Wikipedia (CC BY-SA 4.0)",
    )
    cache = cache_dir / "wikipedia_design.json"
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
        if verbose:
            print(f"  loaded {len(blob['pages'])} Wikipedia articles from cache")
    else:
        pages: list[dict] = []
        missing: list[str] = []
        for i, title in enumerate(titles, 1):
            params = {
                "action": "query", "format": "json", "formatversion": "2",
                "prop": "extracts|revisions", "explaintext": "1",
                "rvprop": "ids|timestamp", "redirects": "1", "titles": title,
            }
            url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
            try:
                data = json.loads(_fetch(url).decode("utf-8"))
            except (urllib.error.URLError, TimeoutError) as exc:
                raise RuntimeError(
                    f"Wikipedia fetch failed on {title!r}: {exc}. Re-run when the network is "
                    f"available, or use --corpus design_permissive to skip Wikipedia entirely. "
                    f"No substitute data is used."
                ) from exc

            for page in data.get("query", {}).get("pages", []):
                if page.get("missing") or not page.get("extract"):
                    missing.append(page.get("title", title))
                    continue
                rev = (page.get("revisions") or [{}])[0]
                pages.append({
                    "title": page["title"],
                    "pageid": page.get("pageid"),
                    "revid": rev.get("revid"),
                    "timestamp": rev.get("timestamp"),
                    "extract": page["extract"],
                })
            if verbose and i % 25 == 0:
                print(f"    {i}/{len(titles)} requested, {len(pages)} retrieved")
            time.sleep(0.15)

        # A large miss rate almost always means the API contract changed, not that 200 curated
        # articles vanished. Fail loudly rather than training on a fraction of the intended corpus.
        if len(pages) < len(titles) * 0.5:
            raise RuntimeError(
                f"only {len(pages)}/{len(titles)} Wikipedia articles returned text. That is a "
                f"fetching problem, not a content problem — refusing to build a corpus from it. "
                f"First few misses: {missing[:8]}"
            )
        if verbose and missing:
            print(f"    {len(missing)} titles had no extract (renamed or removed): {missing[:6]}")

        blob = {"pages": pages, "missing": missing,
                "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(blob), encoding="utf-8")

    parts = []
    for page in blob["pages"]:
        body = clean_wikipedia_extract(page["extract"])
        if len(body) < 400:
            continue
        # Keep the title as a heading: it is a real structural cue and gives the model something to
        # condition on, which makes prompting by topic work at all.
        parts.append(f"# {page['title']}\n\n{body}")
        spec.sources.append({
            "kind": "wikipedia",
            "title": page["title"],
            "pageid": page["pageid"],
            "revid": page["revid"],
            "revision_url": f"https://en.wikipedia.org/w/index.php?oldid={page['revid']}",
            "timestamp": page["timestamp"],
            "licence": "CC BY-SA 4.0",
            "chars": len(body),
        })

    if parts:
        spec.segments = [("wikipedia", "\n\n".join(parts))]
    if verbose:
        print(f"  {len(spec.sources)} articles, {len(spec.text):,} chars")
    return spec


_WIKI_DROP_SECTIONS = {
    "see also", "references", "further reading", "external links", "notes", "citations",
    "bibliography", "sources", "footnotes", "gallery",
}


def clean_wikipedia_extract(text: str) -> str:
    """Drop navigational sections and normalise whitespace.

    Reference lists and "See also" are lists of names and URLs. They are a large fraction of a
    Wikipedia article's characters and contain almost no prose, so training on them teaches the model
    to emit citation fragments. Dropping them raises the useful-token density substantially.
    """
    out_lines: list[str] = []
    skipping = False
    for line in text.split("\n"):
        heading = re.fullmatch(r"\s*=+\s*(.+?)\s*=+\s*", line)
        if heading:
            skipping = heading.group(1).strip().lower() in _WIKI_DROP_SECTIONS
            if not skipping:
                out_lines.append("")
                out_lines.append(f"## {heading.group(1).strip()}")
                out_lines.append("")
            continue
        if not skipping:
            out_lines.append(line)
    body = "\n".join(out_lines)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


# --------------------------------------------------------------------------------------------
# Open textbook: modern graphic design, CC BY 4.0
# --------------------------------------------------------------------------------------------

OPEN_TEXTBOOKS = [
    {
        "base": "https://opentextbc.ca/graphicdesign",
        "title": "Graphic Design and Print Production Fundamentals",
        "author": "Graphic Communications Open Textbook Collective (Ken Jeffery et al.)",
        "year": 2015,
        "publisher": "BCcampus Open Education",
        "licence": "CC BY 4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": (
            "Graphic Design and Print Production Fundamentals by the Graphic Communications Open "
            "Textbook Collective is used under a CC BY 4.0 Licence."
        ),
    },
]

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style|nav|footer|header|form)\b.*?</\1>", re.I | re.S)
_ENTITY = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&apos;": "'",
    "&nbsp;": " ", "&mdash;": "—", "&ndash;": "–", "&hellip;": "…", "&rsquo;": "\u2019",
    "&lsquo;": "\u2018", "&ldquo;": "\u201c", "&rdquo;": "\u201d",
}


# Exact-match site chrome that appears on every Pressbooks page. Left in, BPE would spend merges on
# "Skip to content" and the model would learn to emit navigation — the first extraction attempt
# produced exactly that, which is why this list exists rather than a vague "clean the HTML" comment.
_CHROME_LINES = {
    "skip to content", "previous", "next", "previous/next navigation", "toggle menu",
    "book contents navigation", "contents", "search in book:", "search", "menu",
    "download this book", "share this book", "powered by pressbooks",
    "guides and tutorials", "pressbooks directory", "contact",
    "licence", "license", "share", "print", "email", "twitter", "facebook",
}
_CHROME_PREFIXES = (
    "graphic design and print production fundamentals copyright ©",
    "this book is licensed under a creative commons",
    "except where otherwise noted, this book is",
    "an interactive h5p element has been excluded",
    "you can view it online here:",
)


def html_to_text(html: str) -> str:
    """Minimal HTML → text for Pressbooks chapter pages.

    A full parser is not warranted: Pressbooks emits clean, predictable markup, and pulling in a
    dependency for a handful of regexes would be worse. What *is* warranted is being careful about
    which region of the page is extracted — the first version fell back to the entire document when
    its selector missed, so chapter text arrived wrapped in the page title and navigation menus.
    """
    # Drop <head> outright: it contains the <title>, which otherwise becomes the first line of every
    # chapter ("1.1 Introduction – Graphic Design and Print Production Fundamentals").
    html = re.sub(r"<head\b.*?</head>", " ", html, flags=re.I | re.S)
    html = _SCRIPT_RE.sub(" ", html)

    # Try progressively broader containers. Pressbooks themes vary, so a single selector is fragile.
    text = None
    for pattern in (
        r'<div[^>]+class="[^"]*\bentry-content\b[^"]*"[^>]*>(.*)',
        r'<section[^>]+class="[^"]*\bchapter\b[^"]*"[^>]*>(.*)',
        r'<div[^>]+class="[^"]*\bchapter\b[^"]*"[^>]*>(.*)',
        r"<main\b[^>]*>(.*)",
        r'<div[^>]+id="content"[^>]*>(.*)',
    ):
        m = re.search(pattern, html, re.I | re.S)
        if m:
            text = m.group(1)
            break
    if text is None:
        text = html

    # Cut the footer region if present; everything after it is licence and navigation boilerplate.
    text = re.split(r'<(?:footer|nav)\b', text, maxsplit=1, flags=re.I)[0]

    # Convert block boundaries to paragraph breaks BEFORE stripping tags, or the chapter collapses
    # into one run-on line and every paragraph boundary is lost.
    text = re.sub(r"</(p|div|h[1-6]|li|tr|blockquote|figcaption)>", "\n\n", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = _TAG_RE.sub("", text)
    for k, v in _ENTITY.items():
        text = text.replace(k, v)
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)

    kept: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            kept.append("")
            continue
        low = line.lower()
        if low in _CHROME_LINES or any(low.startswith(p) for p in _CHROME_PREFIXES):
            continue
        kept.append(line)

    return normalise_whitespace("\n".join(kept))


def load_open_textbook(cache_dir: Path, *, verbose: bool = True) -> CorpusSpec:
    """Fetch openly licensed modern design textbooks chapter by chapter.

    Why this is the best source in the corpus for the stated goal
    ------------------------------------------------------------
    The public-domain books are excellent on typography and craft but they are ~1900s printing prose:
    they cannot teach "grid system", "brand identity", "colour management", "prepress workflow", or
    anything about screens. Wikipedia supplies modern vocabulary but in an encyclopedic register that
    jumps topic every article — hard for a small model to model coherently.

    An open textbook is the best of both: modern, coherent, written as continuous instructional prose
    about design as it is actually practised, and licensed **CC BY 4.0** — attribution only, with no
    share-alike obligation, which removes the one legal ambiguity Wikipedia introduces.
    """
    spec = CorpusSpec(
        name="design_textbook",
        description="Openly licensed modern graphic-design textbooks (CC BY 4.0)",
    )
    parts: list[str] = []

    for book in OPEN_TEXTBOOKS:
        slug = book["base"].rstrip("/").rsplit("/", 1)[-1]
        cache = cache_dir / f"textbook_{slug}.json"
        if cache.exists():
            chapters = json.loads(cache.read_text(encoding="utf-8"))
            if verbose:
                print(f"  loaded {len(chapters)} chapters of {book['title']!r} from cache")
        else:
            if verbose:
                print(f"  fetching {book['title']!r} ({book['licence']})")
            urls = _discover_pressbooks_chapters(book["base"], verbose=verbose)
            if not urls:
                if verbose:
                    print("    no chapters discovered — skipping this book")
                continue
            chapters = []
            for i, url in enumerate(urls, 1):
                try:
                    html = _fetch(url).decode("utf-8", errors="replace")
                except (urllib.error.URLError, TimeoutError) as exc:
                    if verbose:
                        print(f"    skip {url}: {exc.__class__.__name__}")
                    continue
                text = html_to_text(html)
                if len(text) > 800:
                    chapters.append({"url": url, "text": text})
                if verbose and i % 10 == 0:
                    print(f"    {i}/{len(urls)} chapters")
                time.sleep(0.3)
            cache.write_text(json.dumps(chapters), encoding="utf-8")

        body = "\n\n".join(c["text"] for c in chapters)
        if len(body) < 5000:
            if verbose:
                print(f"    only {len(body)} chars — skipping")
            continue
        parts.append(body)
        spec.sources.append({
            "kind": "open_textbook",
            **{k: book[k] for k in ("title", "author", "year", "publisher", "licence",
                                    "licence_url", "attribution")},
            "url": book["base"],
            "n_chapters": len(chapters),
            "chars": len(body),
        })
        if verbose:
            print(f"    {len(body):>9,} chars across {len(chapters)} chapters")

    if parts:
        spec.segments = [("open_textbook", "\n\n".join(parts))]
    return spec


def _discover_pressbooks_chapters(base: str, *, verbose: bool = True) -> list[str]:
    """Find chapter URLs, preferring the Pressbooks REST API and falling back to the front page."""
    # Preferred: the documented REST endpoint. Cleaner and stable across theme changes.
    for endpoint in (f"{base}/wp-json/pressbooks/v2/chapters?per_page=100",
                     f"{base}/wp-json/wp/v2/chapter?per_page=100"):
        try:
            data = json.loads(_fetch(endpoint).decode("utf-8"))
            urls = [item["link"] for item in data if isinstance(item, dict) and item.get("link")]
            if urls:
                if verbose:
                    print(f"    {len(urls)} chapters via REST API")
                return urls
        except Exception:  # noqa: BLE001  -- any failure just means fall through to scraping
            continue

    # Fallback: pull chapter links out of the table of contents.
    try:
        html = _fetch(base + "/").decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError):
        return []
    found = re.findall(rf'href="({re.escape(base)}/(?:chapter|part|front-matter|back-matter)/[^"#?]+)"',
                       html)
    urls = sorted(set(found))
    if verbose:
        print(f"    {len(urls)} chapters via table-of-contents scrape")
    return urls


# --------------------------------------------------------------------------------------------
# TinyShakespeare (kept as a control corpus)
# --------------------------------------------------------------------------------------------

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_shakespeare(cache_dir: Path, *, verbose: bool = True) -> CorpusSpec:
    cache = cache_dir / "tinyshakespeare.txt"
    if cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        if verbose:
            print("  fetching TinyShakespeare")
        try:
            text = _fetch(TINY_SHAKESPEARE_URL).decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"could not download TinyShakespeare: {exc}") from exc
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")

    return CorpusSpec(
        name="shakespeare",
        description="TinyShakespeare — a control corpus in a very different register",
        sources=[{
            "kind": "tinyshakespeare",
            "url": TINY_SHAKESPEARE_URL,
            "licence": "Shakespeare: public domain. Compilation: MIT (karpathy/char-rnn).",
            "chars": len(text),
        }],
        segments=[("tinyshakespeare", text)],
    )


# --------------------------------------------------------------------------------------------

CORPORA = {
    "design": "Open textbook (CC BY 4.0) + public-domain typography books + Wikipedia (CC BY-SA 4.0)",
    "design_permissive": "Open textbook (CC BY 4.0) + public-domain books only — NO share-alike content",
    "design_textbook": "Openly licensed modern design textbooks only (CC BY 4.0)",
    "design_pd": "Public-domain typography and printing books only",
    "design_wiki": "Curated Wikipedia graphic-design articles only (CC BY-SA 4.0)",
    "shakespeare": "TinyShakespeare, as a control corpus in a different register",
}


def _combine(name: str, description: str, specs: list[CorpusSpec]) -> CorpusSpec:
    """Merge sources, keeping segments separate so the split stays per-source."""
    specs = [s for s in specs if s.segments]
    if not specs:
        raise RuntimeError(f"corpus {name!r}: every source came back empty; refusing to train on nothing")
    return CorpusSpec(
        name=name,
        description=description,
        sources=[src for s in specs for src in s.sources],
        segments=[seg for s in specs for seg in s.segments],
    )


def load_corpus(name: str, cache_dir: str | Path = "data/p2", *, verbose: bool = True) -> CorpusSpec:
    """Assemble a named corpus, caching every fetch and recording provenance.

    Source ordering within `design` puts the modern textbook first, then the public-domain books, then
    Wikipedia. Order does not affect the split (that is per-segment) but it does affect which register
    the model sees first in each epoch, and starting from coherent modern instructional prose gives
    better conditioning behaviour than starting from encyclopedia stubs.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if name == "shakespeare":
        return load_shakespeare(cache_dir, verbose=verbose)
    if name == "design_pd":
        return load_gutenberg(cache_dir, verbose=verbose)
    if name == "design_wiki":
        return load_wikipedia(cache_dir, verbose=verbose)
    if name == "design_textbook":
        return load_open_textbook(cache_dir, verbose=verbose)

    if name == "design_permissive":
        return _combine(
            "design_permissive",
            "Modern open design textbook (CC BY 4.0) plus public-domain typography books. "
            "Contains no share-alike-licensed text.",
            [load_open_textbook(cache_dir, verbose=verbose),
             load_gutenberg(cache_dir, verbose=verbose)],
        )

    if name == "design":
        return _combine(
            "design",
            "Graphic design: modern open textbook (CC BY 4.0), public-domain typography and "
            "printing books (Project Gutenberg), and curated Wikipedia articles (CC BY-SA 4.0)",
            [load_open_textbook(cache_dir, verbose=verbose),
             load_gutenberg(cache_dir, verbose=verbose),
             load_wikipedia(cache_dir, verbose=verbose)],
        )

    raise ValueError(f"unknown corpus {name!r}. Available: {sorted(CORPORA)}")
