import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from strands import Agent
from strands.models.ollama import OllamaModel


RIOT_UPDATES_URL = (
    "https://www.leagueoflegends.com/en-us/news/game-updates/"
)
OUTPUT_FILE = Path("latest_patch.md")
MODEL_ID = "llama3.2:3b"
MAX_SOURCE_CHARACTERS = 24_000

PATCH_URL_PATTERN = re.compile(
    r"/en-us/news/game-updates/"
    r"(?:league-of-legends-)?patch-(\d+)-(\d+)-notes/?$",
    re.IGNORECASE,
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 Never-Your-Fault-Agent/1.0"
}


def fetch_page(url: str) -> BeautifulSoup:
    """Download and parse a webpage."""
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def find_latest_patch() -> tuple[str, str]:
    """Find the highest-numbered patch on Riot's update page."""
    soup = fetch_page(RIOT_UPDATES_URL)
    patches: list[tuple[tuple[int, int], str]] = []

    for link in soup.select("a[href]"):
        patch_url = urljoin(RIOT_UPDATES_URL, link["href"])
        parsed_url = urlparse(patch_url)

        if parsed_url.hostname not in {
            "leagueoflegends.com",
            "www.leagueoflegends.com",
        }:
            continue

        match = PATCH_URL_PATTERN.fullmatch(parsed_url.path)

        if match:
            version = (
                int(match.group(1)),
                int(match.group(2)),
            )
            patches.append((version, patch_url))

    if not patches:
        raise RuntimeError(
            "No official League patch-notes page was found."
        )

    version, patch_url = max(patches, key=lambda patch: patch[0])
    patch_number = f"{version[0]}.{version[1]}"

    return patch_number, patch_url


def extract_publication_date(soup: BeautifulSoup) -> str:
    """Extract the publication date from the patch page."""
    date_element = (
        soup.select_one('meta[property="article:published_time"]')
        or soup.select_one("time[datetime]")
    )

    if date_element is None:
        raise RuntimeError(
            "The patch publication date could not be verified."
        )

    raw_date = (
        date_element.get("content")
        or date_element.get("datetime")
    )

    if not raw_date:
        raise RuntimeError(
            "The patch publication date was empty."
        )

    try:
        parsed_date = datetime.fromisoformat(
            str(raw_date).replace("Z", "+00:00")
        )

        return (
            f"{parsed_date.strftime('%B')} "
            f"{parsed_date.day}, {parsed_date.year}"
        )
    except ValueError:
        return str(raw_date)


def extract_patch_text(soup: BeautifulSoup) -> str:
    """Extract readable text from the patch article."""
    for element in soup.select(
        "script, style, nav, footer, noscript, svg"
    ):
        element.decompose()

    article = soup.find("article") or soup.find("main") or soup.body

    if article is None:
        raise RuntimeError(
            "The patch article content could not be found."
        )

    lines = []

    for element in article.find_all(
        ["h1", "h2", "h3", "p", "li"]
    ):
        text = " ".join(element.stripped_strings)

        if text:
            lines.append(text)

    patch_text = "\n".join(lines)

    if len(patch_text) < 500:
        raise RuntimeError(
            "The patch page did not contain enough readable text."
        )

    return patch_text[:MAX_SOURCE_CHARACTERS]


def summarize_patch(patch_text: str) -> str:
    """Summarize verified patch text with the local Ollama model."""
    model = OllamaModel(
        host="http://localhost:11434",
        model_id=MODEL_ID,
        max_tokens=1800,
        temperature=0.1,
        keep_alive="10m",
        options={
            "num_ctx": 12_288,
        },
    )

    agent = Agent(
        model=model,
        callback_handler=None,
        system_prompt=(
            "You summarize official League of Legends patch notes. "
            "Use only the supplied text and never invent changes."
        ),
    )

    prompt = f"""
Create a detailed, player-friendly Markdown digest of the official
League of Legends patch notes below.

Use this exact structure:

## Patch Overview

Write 3 to 5 sentences explaining the patch's overall direction and
the most important gameplay effects.

## Most Important Champion Changes

Use these subsections when relevant:

### Buffs

### Nerfs

### Adjustments

List up to 10 champion changes total.

Use this format:

- **Champion**: Explain what changed and its practical gameplay effect
  in one or two concise sentences.

## Important Item Changes

List up to 6 meaningful item changes.

Use this format:

- **Item**: **Buff**, **Nerf**, or **Adjustment** — Explain what changed
  and how it affects players.

## Map, Rune, and System Changes

List up to 6 important changes involving maps, objectives, runes,
matchmaking, progression, game modes, or global systems.

## Bug Fixes That Affect Gameplay

List up to 5 bug fixes only when they could noticeably affect a match.
Do not include minor visual or cosmetic fixes.

## What Players Should Know

Provide exactly 4 concise takeaways describing what players are most
likely to notice in their next matches.

Requirements:

- Use only information explicitly present in the supplied patch notes.
- Keep champion, item, ability, rune, and system names exactly as written.
- Never combine two different champions or items into one entry.
- Do not create new abilities, effects, statistics, or damage types.
- Preserve numerical details when they clarify an important change.
- Prioritize meaningful gameplay changes over minor edits.
- Exclude skins, chromas, cosmetics, esports, promotions, and lore.
- Do not include the patch number, publication date, or source URL.
- Do not speculate about the future metagame.
- Return only the Markdown report.
- Keep the report between approximately 500 and 750 words.
- Stop immediately after the fourth player takeaway.

PATCH NOTES:

{patch_text}
"""

    summary = str(agent(prompt)).strip()

    if not summary:
        raise RuntimeError(
            "The model returned an empty patch summary."
        )

    return summary


def write_report(
    patch_number: str,
    publication_date: str,
    patch_url: str,
    summary: str,
) -> None:
    """Save the final Markdown report."""
    report = f"""# Latest League of Legends Patch

**Patch:** {patch_number}

**Published:** {publication_date}

{summary}

**Source:** {patch_url}
"""

    OUTPUT_FILE.write_text(report, encoding="utf-8")


def main() -> None:
    """Find, summarize, and save the latest Riot patch."""
    try:
        patch_number, patch_url = find_latest_patch()
        patch_page = fetch_page(patch_url)

        publication_date = extract_publication_date(patch_page)
        patch_text = extract_patch_text(patch_page)
        summary = summarize_patch(patch_text)

        write_report(
            patch_number,
            publication_date,
            patch_url,
            summary,
        )

    except (requests.RequestException, RuntimeError) as error:
        raise SystemExit(
            f"Patch update failed: {error}"
        ) from error

    print(
        f"Saved Patch {patch_number} to "
        f"{OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()