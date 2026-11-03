"""A subject is a top-level folder, however deeply the files are nested."""
import os, shutil, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train as T

root = tempfile.mkdtemp()
# a corpus that has grown subdirectories, as a large one must
paths = []
for sub, inner in (("chess", [""]), ("wikipedia", ["A", "B", "C"]),
                   ("code", ["A", "B"])):
    for i in inner:
        d = os.path.join(root, sub, i) if i else os.path.join(root, sub)
        os.makedirs(d, exist_ok=True)
        for k in range(3):
            p = os.path.join(d, f"f{k}.txt")
            open(p, "w").write("x" * 100)
            paths.append(p)
# and a file lying directly in the root
open(os.path.join(root, "loose.txt"), "w").write("x" * 100)
paths.append(os.path.join(root, "loose.txt"))

lanes = {l.name: len(l.paths) for l in T._lanes(paths, 0, [root])}
print("  subjects:", {k: v for k, v in sorted(lanes.items())})
assert lanes["chess"] == 3, lanes
assert lanes["wikipedia"] == 9, "nested folders must not become subjects"
assert lanes["code"] == 6, lanes
assert lanes[os.path.basename(root)] == 1, "a loose file belongs to the root"
assert len(lanes) == 4, f"expected 4 subjects, got {len(lanes)}"

# grouping by the immediate parent - the old rule - would have given many more
byparent = len({os.path.dirname(p) for p in paths})
print(f"  grouping by the parent folder instead would give {byparent} subjects")
assert byparent > len(lanes)
shutil.rmtree(root)
print("  PASS")
