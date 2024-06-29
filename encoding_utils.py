import sys
import locale
import io

def setup_encoding():
    """
    Set up the encoding for the standard output and error streams.

    This function attempts to set the encoding of the standard output and error
    streams to UTF-8. If the encoding cannot be set to UTF-8, it falls back
    to the system's default encoding.

    Parameters:
        None

    Returns:
        None
    """
    try:
        # Try to use UTF-8
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except AttributeError:
        # If that fails, fall back to the system's default encoding
        pass

def get_encoding():
    """
    Returns the preferred encoding for the current system.

    This function attempts to get the preferred encoding by calling the `locale.getpreferredencoding()` method.
    If the method call is successful, the preferred encoding is returned.
    If an exception is raised, the function returns the default encoding 'utf-8'.

    Returns:
        str: The preferred encoding for the current system.

    """
    try:
        return locale.getpreferredencoding()
    except:
        return 'utf-8'


def terminal_supports_unicode():
    """
    Check if the terminal supports Unicode.

    This function uses the `curses` module to check if the terminal supports Unicode. It sets up the terminal
    and retrieves the number of colors supported. If the number of colors is greater than 2, it indicates that
    the terminal supports Unicode.

    Returns:
        bool: True if the terminal supports Unicode, False otherwise.
    """
    import curses

    curses.setupterm()
    return curses.tigetnum("colors") > 2


def safe_encode(text):
    """
    Safely encodes a given text to the specified encoding.

    Args:
        text (str): The text to be encoded.

    Returns:
        str: The encoded text.

    Raises:
        UnicodeEncodeError: If the text cannot be encoded to the specified encoding.

    This function attempts to encode the given text to the encoding returned by the
    `get_encoding()` function. If the encoding is not available, it falls back
    to the 'utf-8' encoding. If the text cannot be encoded to the specified encoding,
    it replaces any invalid characters with the 'REPLACEMENT CHARACTER' (U+FFFD).

    Example:
        >>> safe_encode("Hello, world!")
        'Hello, world!'
        >>> safe_encode("Hello, world! 😊")
        'Hello, world! 😊'
    """
    encoding = get_encoding()
    try:
        return text.encode(encoding).decode(encoding)
    except UnicodeEncodeError:
        return text.encode('utf-8', errors='replace').decode('utf-8')

def safe_print(text):
    print(safe_encode(text))
