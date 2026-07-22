from typing import Tuple
import random, string

def bug_delete(word: str) -> str:
    res = word
    if len(word) <= 2:
        return res
    point = random.randint(1, len(word) - 2)
    res = res[0:point] + res[point + 1:]
    return res

def bug_swap(word: str) -> str:
    if len(word) <= 4:
        return word
    res = word
    points = random.sample(range(1, len(word) - 1), 2)
    a = points[0]
    b = points[1]
    res = list(res)
    w = res[a]
    res[a] = res[b]
    res[b] = w
    res = ''.join(res)
    return res

def get_key_neighbors():
    neighbors = {'q': 'was', 'w': 'qeasd', 'e': 'wrsdf', 'r': 'etdfg', 't': 'ryfgh', 'y': 'tughj', 'u': 'yihjk', 'i': 'uojkl', 'o': 'ipkl', 'p': 'ol', 'a': 'qwszx', 's': 'qweadzx', 'd': 'wersfxc', 'f': 'ertdgcv', 'g': 'rtyfhvb', 'h': 'tyugjbn', 'j': 'yuihknm', 'k': 'uiojlm', 'l': 'opk', 'z': 'asx', 'x': 'sdzc', 'c': 'dfxv', 'v': 'fgcb', 'b': 'ghvn', 'n': 'hjbm', 'm': 'jkn'}
    neighbors['i'] += '1'
    neighbors['l'] += '1'
    neighbors['z'] += '2'
    neighbors['e'] += '3'
    neighbors['a'] += '4'
    neighbors['s'] += '5'
    neighbors['g'] += '6'
    neighbors['b'] += '8'
    neighbors['g'] += '9'
    neighbors['q'] += '9'
    neighbors['o'] += '0'
    return neighbors

def bug_sub_C(word: str) -> str:
    res = word
    key_neighbors = get_key_neighbors()
    point = random.randint(0, len(word) - 1)
    if word[point].lower() not in key_neighbors:
        return word
    choices = key_neighbors[word[point].lower()]
    subbed_choice = choices[random.randint(0, len(choices) - 1)]
    res = list(res)
    res[point] = subbed_choice
    res = ''.join(res)
    return res

def bug_insert(word: str) -> str:
    if len(word) >= 6:
        return word
    res = word
    point = random.randint(1, len(word) - 1)
    res = res[0:point] + random.choice(string.ascii_lowercase) + res[point:]
    return res

def get_bug(word: str):
    bugs = [word]
    if len(word) <= 2:
        return bugs
    bugs.append(bug_delete(word))
    bugs.append(bug_swap(word))
    bugs.append(bug_sub_C(word))
    bugs.append(bug_delete(word))
    bugs.append(bug_swap(word))
    bugs.append(bug_sub_C(word))
    bugs.append(bug_delete(word))
    bugs.append(bug_swap(word))
    bugs.append(bug_sub_C(word))
    return list(set(bugs))

def apply_keyboard_typo(text: str, target: str='really', **_) -> Tuple[str, bool]:
    if not target or target not in text:
        return (text, False)
    candidates = [w for w in get_bug(target) if w != target]
    if not candidates:
        return (text, False)
    new_tok = random.choice(candidates)
    return (text.replace(target, new_tok, 1), True)
