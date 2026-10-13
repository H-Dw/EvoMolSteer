"""Bounded syntax diagnostics only: never repair or return partial JSON values."""
import json


def pointer(path):
    return ''.join('/' + str(v).replace('~', '~0').replace('/', '~1') for v in path)


def container_diagnostic(text):
    """Locate bracket defects while respecting strings/escapes and field paths."""
    stack, index = [], 0

    def value_path():
        if not stack:
            return []
        frame = stack[-1]
        return frame['path'] + [frame['key'] if frame['kind'] == '{' else frame['index']]

    def consumed():
        if stack:
            stack[-1]['state'] = 'comma'
            if stack[-1]['kind'] == '[':
                stack[-1]['index'] += 1

    def issue(frame, actual, position, kind):
        return {'kind': kind, 'container_pointer': pointer(frame['path']),
                'expected_closing': '}' if frame['kind'] == '{' else ']',
                'actual_closing': actual, 'character_offset': position,
                'opening_offset': frame['opening'],
                'instruction': 'Close this container with its expected delimiter before closing its parent. '
                               'Return a complete JSON object; preserve substantive content.'}

    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
        elif char == '"':
            try:
                token, end = json.decoder.scanstring(text, index + 1, True)
            except (ValueError, json.JSONDecodeError):
                return {'kind': 'unterminated_or_invalid_string', 'container_pointer': pointer(value_path()),
                        'character_offset': index, 'instruction': 'Return valid escaped JSON strings and a complete object.'}
            if stack and stack[-1]['kind'] == '{' and stack[-1]['state'] == 'key':
                stack[-1].update(key=token, state='colon')
            else:
                consumed()
            index = end
        elif char in '[{':
            path = value_path()
            consumed()
            stack.append({'kind': char, 'opening': index, 'path': path,
                          'key': None, 'index': 0, 'state': 'key' if char == '{' else 'value'})
            index += 1
        elif char in ']}':
            if not stack:
                return {'kind': 'unexpected_closer', 'container_pointer': '', 'actual_closing': char,
                        'character_offset': index, 'instruction': 'Return exactly one JSON object without trailing delimiters.'}
            if (stack[-1]['kind'], char) not in [('{', '}'), ('[', ']')]:
                return issue(stack[-1], char, index, 'mismatched_closer')
            stack.pop()
            index += 1
        elif char == ':':
            if stack:
                stack[-1]['state'] = 'value'
            index += 1
        elif char == ',':
            if stack:
                stack[-1]['state'] = 'key' if stack[-1]['kind'] == '{' else 'value'
            index += 1
        else:
            consumed()
            end = index + 1
            while end < len(text) and text[end] not in ',]}{["' and not text[end].isspace():
                end += 1
            index = end
    return issue(stack[-1], None, len(text), 'unclosed_container') if stack else None
