#!/usr/bin/env python3
"""Analyze USB bulk hex logs for Uniden R8 handshake tokens and frames.

Input format:
- Text file(s) with one hex string per line (no spaces), typically exported from
  Wireshark/tshark via `-T fields -e usb.capdata` for bulk transfers.
- Supply OUT (host->device) and IN (device->host) files separately so we can
  infer token direction.

Examples:
- python3 -m r8_tools.usb_hex_analyze --out out_hex.txt --in in_hex.txt
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable, List, Tuple


TOKENS = [b"SYN", b"DAT", b"RDY", b"END", b"CLRDAT"]
HEADER_LEN = 16


def _load(path: str) -> List[bytes]:
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            # allow semicolon-delimited
            if ";" in ln:
                ln = ln.split(";")[-1].strip()
            try:
                lines.append(bytes.fromhex(ln))
            except Exception:
                continue
    return lines


def _find_tokens(chunks: List[bytes]) -> List[Tuple[int, bytes]]:
    hits: List[Tuple[int, bytes]] = []
    for i, ch in enumerate(chunks):
        for t in TOKENS:
            if t in ch:
                hits.append((i, t))
    return hits


def _scan_headers(chunks: List[bytes]) -> List[Tuple[int, int, int]]:
    findings = []
    for i, ch in enumerate(chunks):
        data = ch
        for off in range(0, max(0, len(data) - HEADER_LEN + 1)):
            hdr = data[off : off + HEADER_LEN]
            if len(hdr) < HEADER_LEN:
                continue
            verb_len = hdr[4]
            pay_len = int.from_bytes(hdr[2:4], "little")
            if 1 <= verb_len <= 20 and pay_len <= 4096:
                findings.append((i, off, pay_len))
                break
    return findings


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", dest="out_hex")
    ap.add_argument("--in", dest="in_hex")
    args = ap.parse_args(argv)

    out_chunks = _load(args.out_hex) if args.out_hex else []
    in_chunks = _load(args.in_hex) if args.in_hex else []

    print("Host->Device (OUT):", len(out_chunks), "chunks")
    print("Device->Host (IN):", len(in_chunks), "chunks")

    out_tok = _find_tokens(out_chunks)
    in_tok = _find_tokens(in_chunks)
    print("\nToken hits (OUT):", [(i, t.decode('ascii','ignore')) for i, t in out_tok][:10])
    print("Token hits (IN):", [(i, t.decode('ascii','ignore')) for i, t in in_tok][:10])

    in_hdrs = _scan_headers(in_chunks)
    print("\nCandidate BTdn headers (IN):", in_hdrs[:10])

    if in_hdrs:
        i, off, length = in_hdrs[0]
        print(f"\nFirst header at IN chunk {i}, offset {off}, payload_len={length}")
        # Show first few bytes of that chunk
        print("IN chunk hex:", in_chunks[i].hex()[:128], "...")

    # Heuristic: recommend pre-sequence if no device tokens found
    if not in_tok:
        print("\nNo device-issued tokens found. Try pre-sequences RDY,DAT or DAT,RDY, with ACKs.")


if __name__ == "__main__":
    main()

