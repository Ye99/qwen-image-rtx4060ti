"""Interactive Qwen-Image-2.1 client for serve_4060ti.sh: type a prompt, get a PNG.

Prefix a line with options to override defaults, e.g. `seed=7 size=1024x1536 a red fox in snow`
(size is WIDTHxHEIGHT). Line editing via prompt_toolkit: arrows/Home/End/Backspace, Up/Down for
history (persisted across sessions), Ctrl-R to search history, right arrow accepts the grey suggestion.

While an image is generating, Ctrl-C cancels it on the server (frees the GPUs within one denoising
step) and puts the prompt back on the line for editing. At the prompt, Ctrl-C clears the line;
an empty line or Ctrl-D quits.
"""

import random
import re
import sys
import time
from pathlib import Path

import requests
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory

SERVER = "http://127.0.0.1:8000/v1/tasks"
# The server writes the PNG straight here, so client and server must share this filesystem (localhost).
OUT = Path(__file__).resolve().parent / "save_results" / "interactive"
HISTORY = OUT / ".prompt_history"
POLL_SECONDS = 0.5


class Cancelled(Exception):
    pass


def read_prompts():
    """Yield prompts; send() a string back to pre-fill the next prompt with it."""
    if not sys.stdin.isatty():  # piped input: plain line reads, no editor
        for line in sys.stdin:
            yield line.strip()
        return
    session = PromptSession(history=FileHistory(str(HISTORY)), auto_suggest=AutoSuggestFromHistory())
    prefill = ""
    while True:
        try:
            prefill = (yield session.prompt("\nprompt> ", default=prefill).strip()) or ""
        except KeyboardInterrupt:
            prefill = ""
        except EOFError:
            return


def generate(prompt, seed, width, height, path):
    """Submit a task that saves to `path` and wait for it; Ctrl-C cancels the task on the server."""
    body = {"task": "t2i", "prompt": prompt, "seed": seed, "size": [height, width], "save_result_path": str(path)}
    r = requests.post(f"{SERVER}/image/", json=body, timeout=30)
    r.raise_for_status()
    task_id = r.json()["task_id"]
    start = time.time()
    try:
        while True:
            status = requests.get(f"{SERVER}/{task_id}/status", timeout=30).json()
            state = status["status"]
            if state == "completed":
                break
            if state in ("failed", "cancelled"):
                raise RuntimeError(f"task {state}: {status.get('error')}")
            print(f"\r{state} seed={seed}... {time.time() - start:.0f}s (Ctrl-C to cancel)", end="", flush=True)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\rcancelling...", " " * 30, end="", flush=True)
        requests.delete(f"{SERVER}/{task_id}", timeout=30)
        print("\rcancelled; edit the prompt and press Enter to resubmit")
        raise Cancelled from None
    finally:
        print("\r" + " " * 60 + "\r", end="")
    return time.time() - start


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = read_prompts()
    line = next(prompts, "")
    while line:
        raw, prefill = line, ""
        opts = {}
        while m := re.match(r"(seed|size)=(\S+)\s+", line):
            opts[m[1]] = m[2]
            line = line[m.end():]
        try:
            width, height = map(int, opts.get("size", "1024x1024").lower().split("x"))
            seed = int(opts.get("seed", random.randrange(2**31)))
            path = OUT / f"{time.strftime('%Y%m%d-%H%M%S')}_seed{seed}.png"
            seconds = generate(line, seed, width, height, path)
        except ValueError:
            print("bad option: use seed=<int> size=<width>x<height>")
            prefill = raw
        except Cancelled:
            prefill = raw
        except (requests.RequestException, RuntimeError) as e:
            print(f"request failed: {e}")
            prefill = raw
        else:
            path.with_suffix(".txt").write_text(f"{line}\nseed={seed} size={width}x{height}\n")
            print(f"saved {path} ({seconds:.1f}s)")
        try:
            line = prompts.send(prefill)
        except StopIteration:
            break


if __name__ == "__main__":
    main()
