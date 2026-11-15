import json

transcript_path = r"C:\Users\yugpa\.gemini\antigravity\brain\1516d1af-ef44-4a10-ae26-0ff60cb41e15\.system_generated\logs\transcript_full.jsonl"
with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data.get("type") == "USER_INPUT" and data.get("step_index", 0) >= 9:
            print("Found user step:", data.get("step_index"))
            content = data.get("content", "")
            with open(r"E:\New\master_prompt.txt", "w", encoding="utf-8") as out:
                out.write(content)
            print("Wrote master prompt to E:\\New\\master_prompt.txt, len:", len(content))
   
