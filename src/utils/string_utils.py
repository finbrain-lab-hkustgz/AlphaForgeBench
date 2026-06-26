import hashlib

def hash_text_sha256(text: str) -> str:
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()

def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"

def dedent(text: str) -> str:
    """
    Dedent the text and expand the tabs.
    """
    clean = "\n".join(line.strip() for line in text.splitlines())
    return clean
