import sys
import locale
import io

def setup_encoding():
    try:
        # Try to use UTF-8
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except AttributeError:
        # If that fails, fall back to the system's default encoding
        pass

def get_encoding():
    try:
        return locale.getpreferredencoding()
    except:
        return 'utf-8'


def terminal_supports_unicode():
    import curses

    curses.setupterm()
    return curses.tigetnum("colors") > 2


def safe_encode(text):
    encoding = get_encoding()
    try:
        return text.encode(encoding).decode(encoding)
    except UnicodeEncodeError:
        return text.encode('utf-8', errors='replace').decode('utf-8')

def safe_print(text):
    print(safe_encode(text))
