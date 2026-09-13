from llama_cpp import Llama
from huggingface_hub import hf_hub_download

REPO_ID = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
FILENAME = "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

print("Prüfe Modellstatus... (Download startet automatisch, falls nicht vorhanden)")

model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

llm = Llama(
    model_path=model_path,
    chat_format="qwen",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=-1,
    n_batch=512,
    verbose=False
)

messages = [
    {
        "role": "system",
        "content": (
            "Du bist ein direkter KI-Assistent. "
            "Denke nicht laut und gib keine <think>-Abschnitte aus. "
            "Antworte direkt mit dem Ergebnis."
        )
    }
]

print("Chat gestartet. Schreibe 'exit' oder 'quit' zum Beenden.")

try:
    while True:
        user_input = input("\nDu: ").strip()

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Chat beendet.")
            break

        messages.append({"role": "user", "content": user_input})
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=True
        )

        response_parts = []
        pending_content = ""
        in_thinking_block = False
        print("\nAssistent: ", end="", flush=True)
        for chunk in response:
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                pending_content += content

                while pending_content:
                    if in_thinking_block:
                        end_tag = pending_content.find("</think>")
                        if end_tag == -1:
                            pending_content = ""
                            break
                        pending_content = pending_content[end_tag + len("</think>"):]
                        in_thinking_block = False
                    else:
                        start_tag = pending_content.find("<think>")
                        if start_tag != -1:
                            visible_content = pending_content[:start_tag]
                            pending_content = pending_content[start_tag + len("<think>"):]
                            in_thinking_block = True
                        else:
                            safe_length = max(0, len(pending_content) - len("<think>") + 1)
                            visible_content = pending_content[:safe_length]
                            pending_content = pending_content[safe_length:]

                        if visible_content:
                            response_parts.append(visible_content)
                            print(visible_content, end="", flush=True)
                        if in_thinking_block:
                            continue
                        break

        if pending_content and not in_thinking_block:
            response_parts.append(pending_content)
            print(pending_content, end="", flush=True)

        print()
        assistant_message = "".join(response_parts)
        messages.append({"role": "assistant", "content": assistant_message})
except KeyboardInterrupt:
    print("\nChat beendet.")