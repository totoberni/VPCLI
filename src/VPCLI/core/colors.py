def get_contrasting_text_color(hex_bg_color: str) -> str:
    """
    Determines if black or white text is more readable on a given hex background color.

    Args:
        hex_bg_color: The background color as a hex string (e.g., '#RRGGBB').

    Returns:
        'black' or 'white'.
    """
    try:
        # Remove '#' and convert hex to RGB
        hex_color = hex_bg_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        # Calculate luminance using the standard formula
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

        # Return black for light backgrounds, white for dark backgrounds
        return "black" if luminance > 0.5 else "white"
    except (ValueError, IndexError):
        # Fallback for invalid hex codes
        return "white"