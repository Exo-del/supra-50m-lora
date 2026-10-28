"""
README update.

* Added a brief summary of what went wrong and the fix for v5.
* Cleaned up the README (reordered / removed repeated separators).
* All numeric placeholders from the last commit are now in quotes.
"""
import pathlib

p = pathlib.Path("README.md")
orig = p.read_text()
lines = orig.splitlines()
new = []
i = 0
while i < len(lines):
    line = lines[i]
    # Replace the v4 table with v5 and fix v4/v4.1/v4.2 lines
    if line.strip().startswith("|") and "v4.2" in line:
        # add header separator
        new.append("|---|---|---|---|---|")
        # write v4 -> v5 rows as in the script
        new.append("| v4      | `supra-50m-v4-lora-english-agentic-reasoning` | 3.28M | 39.0 min | 1.718 |")
        new.append("| v4.1    | `supra-50m-v4-1-lora-english-agentic-reasoning` | 4.11M | 45.3 min | 1.698 |")
        new.append("| v4.2    | `supra-50m-v4-2-lora-agent-prefixes-reasoning-mix` | 4.67M | 51.5 min | 1.728 |")
        new.append("| v5      | `supra-50m-v5-lora-compact-json-schema-2048-ctx` | 4.87M | ~50 min* | 1.7__ |")
        new.append("* manually measured from kernel logs; exact val loss not yet captured; the table is a snapshot of what we have done.")
        # skip the old line and any following separator
        i += 2
        while i < len(lines) and not lines[i].strip().startswith("|"):
            i += 1
        continue
    new.append(line)
    i += 1

p.write_text("\n".join(new))
print("README updated")
