import re


def make_safe_filename(text):
    # Replace spaces with underscores
    text = text.replace(' ', '_')
    
    # Remove characters that are not allowed in filenames
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    
    # Limit the filename length if necessary (typical limit is 255 characters)
    max_length = 255
    if len(text) > max_length:
        text = text[:max_length]
    
    return text