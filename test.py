
import re

def extract_variables_from_filename(filename, pattern):
    """
    Extract variables from a filename using a specified pattern.

    Parameters:
    - filename (str): The input filename.
    - pattern (str): The pattern with placeholders for variables.

    Returns:
    - dict: A dictionary containing variable names and their values.
    """
    # Escape special characters in the pattern
    pattern = re.escape(pattern)

    # Replace placeholders with capture groups in the pattern
    pattern = re.sub(r'\{(\w+)\}', r'(?P<\1>[^,]+)', pattern)

    # Use the pattern to match the filename
    match = re.match(pattern, filename)

    # If there is a match, return a dictionary of variable names and values
    if match:
        return match.groupdict()
    else:
        return None

# Example usage:
filename = "image_command/123,abc,xyz,123.pt"
pattern = "image_command/{num},{cmd1},{cmd2},{cmd3}.pt"


x = filename.split('/')[-1].split('.')[0].split(",")
n, v, s, th = x
print(type(n))
print(x)