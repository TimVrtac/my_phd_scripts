import pyperclip


def get_one_line():
    multi_line_text = str(pyperclip.paste())
    lines = multi_line_text.split('\r\n')
    one_line_text = ''
    for line_ in lines:
        if one_line_text == '':
            one_line_text += line_
        elif one_line_text[-1] == '-':
            one_line_text += line_
        else:
            one_line_text += ' ' + line_
    pyperclip.copy(one_line_text)
    return one_line_text


print(get_one_line())
