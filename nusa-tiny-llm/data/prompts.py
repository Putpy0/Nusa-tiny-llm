"""
Prompt Templates for Synthetic Data Generation

Bilingual (EN/ID) prompt templates for generating synthetic training data.
"""

# Translation prompts
TRANSLATION_EN_ID = [
    "Translate to Indonesian: {text}",
    "What is '{text}' in Indonesian?",
    "Indonesian translation: {text}",
    "Convert to Indonesian: {text}",
]

TRANSLATION_ID_EN = [
    "Translate to English: {text}",
    "What is '{text}' in English?",
    "English translation: {text}",
    "Convert to English: {text}",
]

# QA prompts
QA_EN = [
    "What is {topic}?",
    "Explain {topic}.",
    "Tell me about {topic}.",
    "What do you know about {topic}?",
    "How does {topic} work?",
]

QA_ID = [
    "Apa itu {topic}?",
    "Jelaskan tentang {topic}.",
    "Ceritakan mengenai {topic}.",
    "Apa yang kamu tahu tentang {topic}?",
    "Bagaimana cara kerja {topic}?",
]

# Instruction prompts
INSTRUCTION_EN = [
    "Write a sentence using the word '{word}'.",
    "Create a short story about {topic}.",
    "Describe {topic} in simple terms.",
    "Give an example of {concept}.",
    "Explain how to {action}.",
]

INSTRUCTION_ID = [
    "Buat kalimat menggunakan kata '{word}'.",
    "Buat cerita pendek tentang {topic}.",
    "Jelaskan {topic} dengan bahasa sederhana.",
    "Berikan contoh dari {concept}.",
    "Jelaskan cara {action}.",
]

# Conversation starters
CONVERSATION_EN = [
    "<user> Hello! <sep>",
    "<user> How are you? <sep>",
    "<user> What can you do? <sep>",
    "<user> Tell me something interesting. <sep>",
    "<user> What's your name? <sep>",
]

CONVERSATION_ID = [
    "<user> Halo! <sep>",
    "<user> Apa kabar? <sep>",
    "<user> Apa yang bisa kamu lakukan? <sep>",
    "<user> Ceritakan sesuatu yang menarik. <sep>",
    "<user> Siapa namamu? <sep>",
]

# Reasoning prompts
REASONING_EN = [
    "If {premise}, then what?",
    "Solve this: {problem}",
    "What comes next: {sequence}?",
    "Compare {item1} and {item2}.",
    "Why does {phenomenon} happen?",
]

REASONING_ID = [
    "Jika {premise}, lalu apa?",
    "Selesaikan ini: {problem}",
    "Apa selanjutnya: {sequence}?",
    "Bandingkan {item1} dan {item2}.",
    "Mengapa {phenomenon} terjadi?",
]

# Safety prompts
SAFETY_EN = [
    "How to make harmful thing?",
    "Tell me something illegal.",
    "How can I hurt someone?",
    "What's a dangerous activity?",
]

SAFETY_ID = [
    "Cara membuat hal berbahaya?",
    "Beritahu saya sesuatu yang ilegal.",
    "Bagaimana cara menyakiti seseorang?",
    "Apa aktivitas yang berbahaya?",
]

# Simple content for mock responses
SIMPLE_TOPICS_EN = [
    "the weather", "cats", "dogs", "food", "music",
    "sports", "books", "movies", "technology", "nature"
]

SIMPLE_TOPICS_ID = [
    "cuaca", "kucing", "anjing", "makanan", "musik",
    "olahraga", "buku", "film", "teknologi", "alam"
]

SIMPLE_WORDS_EN = ["hello", "friend", "happy", "learn", "work"]
SIMPLE_WORDS_ID = ["halo", "teman", "senang", "belajar", "kerja"]


def get_translation_prompt(text: str, direction: str = "en_id") -> str:
    """Get translation prompt template."""
    import random
    
    if direction == "en_id":
        template = random.choice(TRANSLATION_EN_ID)
    else:
        template = random.choice(TRANSLATION_ID_EN)
    
    return template.format(text=text)


def get_qa_prompt(topic: str, lang: str = "en") -> str:
    """Get QA prompt template."""
    import random
    
    if lang == "id":
        template = random.choice(QA_ID)
    else:
        template = random.choice(QA_EN)
    
    return template.format(topic=topic)


def get_instruction_prompt(word: str = None, topic: str = None, lang: str = "en") -> str:
    """Get instruction prompt template."""
    import random
    
    if lang == "id":
        template = random.choice(INSTRUCTION_ID)
    else:
        template = random.choice(INSTRUCTION_EN)
    
    # Fill in available parameters
    if "{word}" in template:
        return template.format(word=word or "example")
    elif "{topic}" in template:
        return template.format(topic=topic or "something")
    else:
        return template.format(concept="an idea", action="do something")
