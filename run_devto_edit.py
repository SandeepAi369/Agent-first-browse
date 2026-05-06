import sys
import asyncio
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "python-orchestrator"))

from advanced_agent import run_agent

objective = """
MISSION: SURGICAL EDIT OF PUBLISHED DEV.TO POST

EXECUTION STEPS:
1. DIRECT NAVIGATION: 
   - Navigate directly to the EXACT URL of the published post: https://dev.to/chidari_sandeep_c8e0478a1/why-i-rewrote-my-90-engine-meta-search-in-rust-41l5
   - Once loaded, locate and click the "Edit" or "Manage" button on the post to open the editor. (If it asks you to edit the article, confirm).

2. SURGICAL URL REPLACEMENT (Ghost Engine):
   - In the markdown body editor, find the existing placeholder GitHub link (e.g., https://github.com/yourusername/searchwala).
   - Use GhostCursor to select it or ghost_type to delete the wrong URL.
   - Insert the EXACT correct URL: https://github.com/SandeepAi369/SearchWala
   - DO NOT change or rewrite any other text in the post body. ONLY fix the URL.

3. SAVE/PUBLISH: 
   - Click the "Save changes" or "Publish" button to commit the edit.

OUTPUT: 
Do not exit until the post is successfully saved with the correct URL. Output action_type="done" when finished.
"""

if __name__ == "__main__":
    print("Starting Live Edit on Dev.to...")
    asyncio.run(run_agent(objective))
