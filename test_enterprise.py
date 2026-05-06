"""Smoke test for all enterprise modules."""
import sys
sys.path.insert(0, "python-orchestrator")

# Test 1: Ghost Input
from ghost_input import ghost_click, ghost_type, ghost_scroll, _generate_bezier_path
path = _generate_bezier_path((0,0), (100,100))
print(f"Ghost Input OK — Bezier path has {len(path)} points")

# Test 2: Campaign Memory
from campaign_memory import CampaignMemory
mem = CampaignMemory()
print(f"Campaign Memory OK — {mem.get_post_count()} posts recorded")

# Test 3: Content Critic
from content_critic import ContentCritic, _scan_jargon
critic = ContentCritic()
hits = _scan_jargon("This cutting-edge solution leverages AI to revolutionize search")
print(f"Content Critic OK — found {len(hits)} jargon hits: {hits}")

# Test 4: Twitter Thread
from twitter_thread import ThreadGenerator
gen = ThreadGenerator()
thread = gen.generate("Test Title", "Test body content about Rust and open source", "https://github.com/test")
print(f"Twitter Thread OK — generated {thread.total_tweets} tweets with tags {thread.hashtags}")

# Test 5: GitHub Engagement
from github_engagement import GitHubEngagement
eng = GitHubEngagement()
targets = eng.select_targets(max_targets=2)
print(f"GitHub Engagement OK — {len(targets)} targets selected")

# Test 6: Worker Planner multi-key
from app.browser_promoter.worker_planner import _build_worker_llms
p, s, d, t = _build_worker_llms()
print(f"Worker Planner OK — NVIDIA={len(p)}, Groq={len(s)}, DeepSeek={len(d)}, Gemma={len(t)}")

print()
print("=== ALL ENTERPRISE MODULES VERIFIED ===")
