"""Paths stored in an assembly XML are relative to the XML file's folder."""
import os


def resolve_xml_path(path, base_dir=None):
    """Resolve a path read from an XML against the XML's folder.

    Absolute paths stay as they are. Backslashes count as separators, unless the path exists
    exactly as written (a filename that contains a backslash on POSIX). A relative path that
    does not exist next to the XML but does exist from the working directory is kept as is
    (older XMLs rely on the working directory).
    """
    if base_dir is None:
        return path
    for candidate in dict.fromkeys((path, path.replace(chr(92), "/"))):
        if os.path.isabs(candidate) or candidate[1:2] == ":":
            return candidate
        resolved = os.path.normpath(os.path.join(base_dir, candidate))
        if os.path.exists(resolved):
            return resolved
    normalized = path.replace(chr(92), "/")
    return normalized if os.path.exists(normalized) else os.path.normpath(os.path.join(base_dir, normalized))


def relativize_xml_path(path, base_dir):
    """Write a path relative to the XML's folder, with forward slashes (absolute if it cannot be)."""
    try:
        return os.path.relpath(path, base_dir).replace("\\", "/")
    except ValueError:  # other drive on Windows
        return os.path.abspath(path).replace("\\", "/")
