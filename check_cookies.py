import json

with open("/home/sandeep/agent first IDE/persistence/session_state.json") as f:
    data = json.load(f)

cookies = data.get("cookies", [])
origins = data.get("origins", [])

print(f"{'='*70}")
print(f"  SESSION STATE REPORT")
print(f"{'='*70}")
print(f"  Total cookies saved: {len(cookies)}")
print(f"  Total origins saved: {len(origins)}")
print()

google = [c for c in cookies if "google" in c.get("domain", "") or "gmail" in c.get("domain", "") or "youtube" in c.get("domain", "")]
print(f"  🔑 Google/Gmail/YouTube cookies: {len(google)}")
for c in google:
    exp = c.get("expires", -1)
    exp_str = "session" if exp == -1 else f"expires={int(exp)}"
    print(f"     {c['name']:30s}  domain={c['domain']:40s}  {exp_str}")

print()
other = [c for c in cookies if c not in google]
print(f"  🍪 Other cookies: {len(other)}")
for c in other:
    print(f"     {c['name']:30s}  domain={c['domain']}")

print()
print(f"{'='*70}")
print(f"  ✅ All cookies are stored at:")
print(f"     /home/sandeep/agent first IDE/persistence/session_state.json")
print(f"{'='*70}")
