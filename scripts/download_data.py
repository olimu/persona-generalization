
#!/usr/bin/env python3
"""Download and install external datasets required by the experiments.

Sycophancy datasets are downloaded from the source repository. Only the five
JSONL files under training_data/sycophancy/ are installed.

Emergent misalignment datasets are downloaded from the upstream encrypted
archive and decrypted with easy-dataset-share. Three upstream datasets are
installed into data/em/. Four project-provided OOD evaluation datasets must
already be present there.

SQuAD, SquadShifts, and XQuAD are handled by scripts/prepare_data.py.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"

SYCOPHANCY_REPO_ZIP = (
    "https://codeload.github.com/"
    "stay1to0/Sycophancy_Emergent_Misalignment_and_Gated_attention_FT/"
    "zip/refs/heads/main"
)

EM_ENCRYPTED_ARCHIVE = (
    "https://raw.githubusercontent.com/jacksonwebster/"
    "emergent-misalignment/main/training_data/training_datasets.zip.enc"
)

SYCOPHANCY_DOMAINS = (
    "medical",
    "financial",
    "legal",
    "security",
    "sport",
)

EM_ARCHIVE_FILES = (
    "bad_medical_advice.jsonl",
    "good_medical_advice.jsonl",
    "extreme_sports.jsonl",
)

EM_CUSTOM_OOD_FILES = (
    "bad_cooking_advice.jsonl",
    "bad_cybersec_advice.jsonl",
    "bad_diy_advice.jsonl",
    "bad_social_advice.jsonl",
)

EM_PASSWORD = "model-organisms-em-datasets"


def download(url: str, destination: Path, refresh: bool = False) -> None:
    """Download a URL unless a non-empty cached file is available."""
    if (
        destination.is_file()
        and destination.stat().st_size > 0
        and not refresh
    ):
        print(f"Using cached download: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "persona-generalization-research/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            try:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)

                if temporary.stat().st_size == 0:
                    raise RuntimeError(f"Downloaded an empty file from {url}")

                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc

    print(f"Downloaded: {destination}")


def safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract a ZIP archive while rejecting path traversal entries."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()

                if target != root and root not in target.parents:
                    raise RuntimeError(
                        f"Unsafe path in ZIP archive {zip_path}: "
                        f"{member.filename}"
                    )

            archive.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Not a valid ZIP archive: {zip_path}") from exc


def extract_nested_zips(directory: Path) -> None:
    """Extract nested ZIP files, including bundled dataset archives."""
    seen: set[Path] = set()

    while True:
        candidates = [
            path
            for path in directory.rglob("*.zip")
            if path not in seen
        ]

        if not candidates:
            return

        for archive_path in candidates:
            seen.add(archive_path)
            destination = (
                archive_path.parent / f"{archive_path.stem}_extracted"
            )

            if destination.exists():
                continue

            destination.mkdir(parents=True)

            try:
                safe_extract(archive_path, destination)
            except RuntimeError:
                # Ignore unrelated or encrypted nested archives.
                shutil.rmtree(destination, ignore_errors=True)


def is_sycophancy_domain_file(path: Path, domain: str) -> bool:
    """Match a domain JSONL only within training_data/sycophancy/."""
    if path.name != f"{domain}.jsonl":
        return False

    return (
        path.parent.name == "sycophancy"
        and "training_data" in path.parts
    )


def download_sycophancy(
    data_dir: Path,
    downloads_dir: Path,
    refresh: bool = False,
) -> None:
    """Download and install the five intended sycophancy datasets."""
    archive_path = downloads_dir / "sycophancy_source.zip"
    extracted_dir = downloads_dir / "sycophancy_source"

    download(SYCOPHANCY_REPO_ZIP, archive_path, refresh=refresh)

    # Re-extract from scratch so stale files cannot affect dataset selection.
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)

    extracted_dir.mkdir(parents=True)
    safe_extract(archive_path, extracted_dir)
    extract_nested_zips(extracted_dir)

    matches: dict[str, list[Path]] = {
        domain: []
        for domain in SYCOPHANCY_DOMAINS
    }

    for path in extracted_dir.rglob("*.jsonl"):
        for domain in SYCOPHANCY_DOMAINS:
            if is_sycophancy_domain_file(path, domain):
                matches[domain].append(path)

    missing = [
        domain
        for domain, candidates in matches.items()
        if not candidates
    ]

    if missing:
        expected_names = {
            f"{domain}.jsonl"
            for domain in SYCOPHANCY_DOMAINS
        }
        found = sorted(
            path.relative_to(extracted_dir).as_posix()
            for path in extracted_dir.rglob("*.jsonl")
            if path.name in expected_names
        )
        found_text = (
            "\n".join(f"  {name}" for name in found[:80])
            if found
            else "  (no matching JSONL filenames found)"
        )

        raise RuntimeError(
            "Could not find all expected datasets under "
            "training_data/sycophancy/.\n"
            f"Missing domains: {', '.join(missing)}\n"
            f"Extracted source: {extracted_dir}\n"
            "Matching filenames found elsewhere in the archive:\n"
            f"{found_text}"
        )

    ambiguous = {
        domain: candidates
        for domain, candidates in matches.items()
        if len(candidates) > 1
    }

    if ambiguous:
        details = []
        for domain, candidates in ambiguous.items():
            details.append(f"{domain}.jsonl:")
            details.extend(
                f"  {candidate.relative_to(extracted_dir)}"
                for candidate in candidates
            )

        raise RuntimeError(
            "Multiple candidates found inside training_data/sycophancy/. "
            "Refusing to choose arbitrarily:\n"
            + "\n".join(details)
        )

    target_dir = data_dir / "sycophancy"
    target_dir.mkdir(parents=True, exist_ok=True)

    for domain, candidates in matches.items():
        source = candidates[0]
        target = target_dir / f"{domain}.jsonl"
        shutil.copy2(source, target)
        print(f"Installed sycophancy dataset: {target}")


def decrypt_em_archive(archive_path: Path, destination: Path) -> None:
    """Decrypt the upstream EM archive into a temporary directory."""
    try:
        subprocess.run(
            [
                "easy-dataset-share",
                "magic-unprotect-dir",
                str(archive_path),
                "-p",
                EM_PASSWORD,
                "--remove-canaries",
            ],
            check=True,
            cwd=destination,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "EM dataset decryption requires easy-dataset-share. Install it "
            'with: python -m pip install "easy-dataset-share==0.1.0"'
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to decrypt the EM archive. Check the installed "
            "easy-dataset-share version and the downloaded archive."
        ) from exc


def download_em_archive(
    data_dir: Path,
    downloads_dir: Path,
    refresh: bool = False,
) -> None:
    """Download, decrypt, and install upstream EM datasets."""
    archive_path = downloads_dir / "training_datasets.zip.enc"
    download(EM_ENCRYPTED_ARCHIVE, archive_path, refresh=refresh)

    em_dir = data_dir / "em"
    em_dir.mkdir(parents=True, exist_ok=True)

    # Decrypt in a temporary directory rather than relying on files left by
    # previous manual extraction attempts.
    with TemporaryDirectory(prefix="em_datasets_") as temporary:
        temporary_dir = Path(temporary)
        temporary_archive = temporary_dir / archive_path.name
        shutil.copy2(archive_path, temporary_archive)

        decrypt_em_archive(temporary_archive, temporary_dir)

        for filename in EM_ARCHIVE_FILES:
            matches = list(temporary_dir.rglob(filename))

            if not matches:
                raise RuntimeError(
                    f"Decrypted EM archive did not contain {filename}."
                )

            if len(matches) > 1:
                raise RuntimeError(
                    f"Found multiple copies of {filename} in the decrypted "
                    "archive. Refusing to select one arbitrarily."
                )

            target = em_dir / filename
            shutil.copy2(matches[0], target)
            print(f"Installed EM dataset: {target}")

    # These OOD evaluation sets are project-provided, not part of the
    # upstream archive. Validate their presence without overwriting them.
    missing = [
        filename
        for filename in EM_CUSTOM_OOD_FILES
        if not (em_dir / filename).is_file()
    ]

    if missing:
        raise RuntimeError(
            f"Missing project-provided OOD datasets in {em_dir}: "
            f"{', '.join(missing)}. Include these files in the release."
        )

    print("All required EM datasets are available.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("em", "sycophancy", "all"),
        default="all",
        help="Which external dataset sources to download.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Data directory (default: <repo>/data).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download source archives again, even if cached.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    downloads_dir = data_dir / "downloads"
    data_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("sycophancy", "all"):
        download_sycophancy(
            data_dir,
            downloads_dir,
            refresh=args.refresh,
        )

    if args.mode in ("em", "all"):
        download_em_archive(
            data_dir,
            downloads_dir,
            refresh=args.refresh,
        )

    print(f"\nDataset download step complete: {data_dir}")


if __name__ == "__main__":
    main()
