"""tokcmp.py tokenizer.json model.air files... - compare air's tokenizer with the HF reference."""
import subprocess, sys, tokenizers
tk = tokenizers.Tokenizer.from_file(sys.argv[1]); bad = 0; total = 0
stress = ["Hello, world! It's   a  test.\n\n\n  x", "日本語のテキストと中文混合 text", "naïve café — résumé ½ ² Ⅻ 12345678",
          "emoji 👍🏽👨‍👩‍👧 done", "\r\n\r\n\tTabs\t\tand  spaces   \n", "I'LL WE'RE they'Ve O'Neil don't", "a/b//c\\d  \n/ ",
          "x = [1,2,3]; // comment\n\tif (a<b) {return a;}", "ÀÉÎõü ÇĞŞ ǅungla ʰmodifier", "١٢٣ ١٢ ٣ arabic digits ܐ", "    "]
def check(name, text):
    global bad, total
    ref = tk.encode(text, add_special_tokens=False).ids
    out = subprocess.run(["./tok", sys.argv[2]], input=text.encode(), capture_output=True).stdout.decode().split()
    got = [int(x) for x in out]; total += 1
    if got != ref:
        bad += 1; i = next((k for k in range(min(len(got), len(ref))) if got[k] != ref[k]), min(len(got), len(ref)))
        print(f"MISMATCH {name}: at {i}: got {got[i:i+5]} ref {ref[i:i+5]} | {tk.decode(ref[max(0,i-3):i+3])!r}")
for i, s in enumerate(stress): check(f"stress{i}", s)
for f in sys.argv[3:]: check(f, open(f, encoding="utf-8", errors="replace").read())
print(f"{total - bad}/{total} identical")
