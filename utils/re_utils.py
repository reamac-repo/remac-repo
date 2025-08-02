import re

def extract_content(content, filter=None):
    """
    extract useful content based on filter
    """
    if filter == 'python':
        match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
    elif filter == 'A:':
        match = re.search(r'A:\s*(.*)', content)
    elif filter == 'Output:':
        match = re.search(r'Output:\s*(.*)', content)
    elif filter == 'Plan:[]':
        match = re.search(r'Plan: \[(.*?)\]', content)
    elif filter == 'result':
        index = content.find("result")
        if index != -1:
            return content[index:].strip()
        else:
            return content.strip()
    elif filter == 'True or False':
        assert 'True' in content or 'False' in content or 'true' in content or 'false' in content
        if 'True' in content or 'true' in content:
            return 'True'
        else:
            return 'False'
    elif filter == '```':
        return re.sub(r'^`+|`+$', '', content.strip()).strip()
    elif filter == '[]':
        pattern = r'\[.*?\]'
        matches = re.findall(pattern, content)
        assert len(matches) > 0, 'No match found'
        return matches[-1]
    else:
        return content.strip()
    
    if match:
        return match.group(1).strip()
    return content.strip()


if __name__ == '__main__':
    content1 = """
```python
plain text
```
"""
    content2 = """
A: plain text
"""
    content3 = """
Output: plain text
"""
    content4 = """
Plan: [text1, text2]
"""
    context5 = """
```python    
result = plain text
```
"""
    context6 = """
plain text True plain text
"""
    content7 = """
```
    plain text
```
"""
    content8 = """
plain text1
[plain text]
plain text2
"""
    print(extract_content(content1, 'python'))
    print(extract_content(content2, 'A:'))
    print(extract_content(content3, 'Output:'))
    print(extract_content(content4, 'Plan:[]'))
    print(extract_content(context5, 'result'))
    print(extract_content(context6, 'True or False'))
    print(extract_content(content7, '```'))
    print(extract_content(content8, '[]'))