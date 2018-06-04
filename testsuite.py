import os
import sys

addpp = __import__('add++')

tests = [

    [
        addpp.Script,
        
        'x:"Hello, World!"\nO',
        [],
        [False, False],
        False,
        False,
        
        'Hello, World!',
        
        'Variable assignment failed',
    ],
    
    [
        addpp.Script,
        
        'D,f,^,"Hello, World!"\n$f\nO',
        [],
        [False, False],
        False,
        False,
        
        'Hello, World!',

        'Functions failed',
    ],

    [
        addpp.Script,
        
        'a:1\n`a\nO',
        [],
        [False, False],
        False,
        False,

        '1',

        'Variable activation failed'
    ],

    [
        addpp.Script,
        
        'a:?\ny:10\na+y\nO',
        [1],
        [False, False],
        False,
        False,

        '11',

        'Infix commands failed'
    ],

    [
        addpp.Script,

        'x:?\nO',
        [10],
        [False, False],
        False,
        False,

        '10',

        'Prefix active commands failed',

    ],

    [
        addpp.Script,

        'a:?\nOa',
        [10],
        [False, False],
        False,
        False,

        '10',

        'Prefix inactive commands failed',

    ],

    [
        addpp.Script,
        
        'D,f,@*,"Hello, World!"',
        ['abc'],
        [True, False],
        False,
        False,

        "['abc' 'Hello, World!']",

        'Implicit functions failed'
    ],

    [
        addpp.Script,
        
        'D,f,@*,"Hello,"\nD,g,@*,"World!"',
        ['abc'],
        [True, 'g'],
        False,
        False,

        "['abc' 'World!']",

        'Specify flag failed'
    ],

    [
        addpp.Script,

        'x:10\ny:2\n]divmod\nO',
        [],
        [False, False],
        False,
        False,

        '(5, 0)',

        'Extensions commands failed'

    ],

    [
        addpp.Script,

        '}print("abc")',
        [],
        [False, False],
        False,
        False,

        'abc',

        'Python exec failed'

    ],

]

for elem in tests:
    if len(elem) != 8:
        print('Invalid test:', *elem, sep = '\n', end = '\n\n')
        sys.exit(0)

results = []

for script, test, inputs, implicit, tokens, debug, ret, reason in tests:
    with addpp.suppress_output('testtemp', False):
        with addpp.suppress_output('testtemperr', True):
            try: script(test, inputs, implicit, tokens, debug)
            except: pass

    with open('testtemp', 'r') as outfile:
        out = outfile.read().strip()
    with open('testtemperr', 'r') as errfile:
        err = errfile.read().strip()

    results.append(out == ret and not err)

os.remove('testtemp')
os.remove('testtemperr')

for index, res in enumerate(results):
    if not res:
        failed = tests[index][1]
        reason = tests[index][7]
        
        print('; Failed test:', failed,
              '; Reason:\n; {}'.format(reason),
              sep = '\n; =-= ;\n', end = '\n\n')
