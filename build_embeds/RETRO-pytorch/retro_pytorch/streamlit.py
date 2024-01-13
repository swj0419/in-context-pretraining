import streamlit.components.v1 as components


def text_with_scrollbar(text, height="450px"):
    components.html(f"""<div style="height:{height}; overflow: scroll;">{text}</div>""", height=500)


DOMAINS = [
    "BookCorpusFair",
    "ccnewsv2",
    "CommonCrawl",
    "DM_Mathematics",
    "Enron_Emails",
    "Gutenberg_PG-19",
    "HackerNews",
    "OpenSubtitles",
    "OpenWebText2",
    "redditflattened",
    "stories",
    "USPTO",
    "Wikipedia_en",
]
FOLDS = ["train", "valid"]
