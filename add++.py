import argparse
import collections
import contextlib
import functools as fn
import itertools as it
import math
import operator as op
import os
import random
import re
import sys

import error
import extensions

GLOBALREGISTER = None
VERSION = 5.8

identity = lambda a: a

INFIX = '+*/\^><%=|RD-'
PREFIX = '!#NROoHhd_B'

DEFAULT = {
    
    '*': False,
    '^': False,
    '?': False,
    ':': False,
    '!': False,
    '~': False,
    '#': False,
    '&': False,
    
    '@': list(),

    'lambda': False,

}

FLAGS = list(DEFAULT.keys())
FLAGS.remove('@')
FLAGS.remove('lambda')
FLAGS = ''.join(FLAGS)

number = re.compile(r'''

    ^
    ((-?[1-9]\d*|0)\.\d+
    |
    (-?[1-9]\d*|0)
    |
    (
        (
            (-?[1-9]\d*|0)
            (\.\d+)?
            [+-]
        )?
	(-?[1-9]\d*|0)
	(\.\d+)?j
    ))$

''', re.VERBOSE)

varassign = re.compile(r'''

    ^
    ([A-Za-z]+)
    :
    (.*)$

''', re.VERBOSE)

varswap = re.compile(r'''
                     
    ^
    ([A-Za-z]+)
    &
    ([A-Za-z]+)
    $

''', re.VERBOSE)

varactive = re.compile(r'''

    ^
    `
    ([A-Za-z]+)
    $

''', re.VERBOSE)

comment = re.compile(r'''

    ^
    \s+
    (.*?)
    (;
    .*)?$

''', re.VERBOSE)

infix = re.compile(r'''

    ^
    ([A-Za-z]+|.*?)?
    ([{}])
    (.*)$

'''.format(INFIX), re.VERBOSE)

prefix = re.compile(r'''

    ^
    ([{}])
    ([A-Za-z]+|.*?)$

'''.format(PREFIX), re.VERBOSE)

construct = re.compile(r'''

    ^(
        ((W|I|D)
         ([A-Za-z]+
          (?:
           [{}]
           [A-Za-z]+)?
         )|
         ([{}]
          ([A-Za-z]+|.*?)))
        |(F([A-Za-z]+|.*?))
        |(E([A-Za-z]+|.*?))
    ),
    (.*)$
    
'''.format(INFIX, PREFIX), re.VERBOSE)

function = re.compile(r'''

    ^
    (?:D,([A-Za-z]+),((?:[@#]{}[A-Za-z]+|[@{}]|[^,])*),(.*)$)
    |
    (?:L((?:[{}]|[^,])*),(.*)$)
    |
    (?:\$(lambda\ \d+)>?(.*)$)
    |
    (?:\$([A-Za-z]+)>?(.*)$)

'''.format('{1,2}', FLAGS, FLAGS), re.VERBOSE)

additionals = re.compile(r'''

    ^
    (
        ]
        [A-Za-z]+
    )|
    (
        }
        .*
    )
    $

''', re.VERBOSE)

class addpp(object):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)

addpp.code_page = '''€§«»Þþ¦¬£\t\nªº\r↑↓¢Ñ×¡¿ß‽⁇⁈⁉ΣΠΩΞΔΛ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~'''

@contextlib.contextmanager
def suppress_output(pipe = os.devnull, error = False):
    with open(pipe, 'w') as output:
        if error:
            old = sys.stderr
            sys.stderr = output
        else:
            old = sys.stdout
            sys.stdout = output
        
        try:  
            yield
        finally:
            if error:
                sys.stderr = old
            else:
                sys.stdout = old

def isdigit(string):
    return all(i in '1234567890-.' for i in string)

def split(string, sep):
    final = []
    splitting = True
    index = 0
    for i, char in enumerate(string):
        if char == '"':
            splitting ^= 1
            continue
        if char == sep and splitting:
            final.append(string[index : i])
            index = i + 1
    return final + [string[index : ]]

def eval_(string):
    if string[0] == '[' and string[-1] == ']':
        string = ', '.join(split(string, ' '))
    try:
        return eval(string)
    except:
        return string

def nest(f, g):
    return lambda x: f(g(x))

def not_none(value):
    return value is not None

def splitting(string, splitter):
    cmds = ['']
    ignore = 0
    skips = 0
    for index, char in enumerate(string):
        if skips:
            skips -= 1
            continue
        
        if char == '"':
            if ignore == 1:
                ignore = 0
            elif ignore == 0:
                ignore = 1
                
        if char == "'":
            if ignore == 2:
                ignore = 0
            elif ignore == 0:
                ignore = 1

        if char == splitter and not ignore:
            if string[index+1] == splitter:
                cmds[-1] += char
                skips = 1
            else:
                cmds.append('')
        else:
            cmds[-1] += char

    return cmds

def whileloop(inputs, var, v, funcs, cond, *run, do = False):
    run = '\n'.join(run)
    
    if do:
        state = Script(run, inputs, vars_ = [var, v, funcs])
        var = state.variables
        v = state.var
        
    while Script(cond, inputs, loop = True, vars_ = [var, v, funcs]).variables['#']:
        state = Script(run, inputs, vars_ = [var, v, funcs])
        var = state.variables
        v = state.var
    return var.copy()

def forloop(inputs, var, v, funcs, itervar, *run):
    run = '\n'.join(run)
    itervar = var[itervar]
    for i in range(1, itervar+1):
        var['i'] = i
        state = Script(run, inputs, vars_ = [var, v, funcs])
        var = state.variables
        v = state.var
        
    var.pop('i', None)
    return var.copy()

def doloop(inputs, var, v, funcs, cond, *run):
    return whileloop(inputs, var, v, funcs, cond, *run, do = True)

def each(inputs, var, v, funcs, itervar, *run):
    run = '\n'.join(run)
    itervar = var[itervar]
    if isinstance(itervar, (int, float)):
        itervar = range(1, int(itervar)+1)

    for i in itervar:
        var['i'] = i
        state = Script(run, inputs, vars_ = [var, v, funcs])
        var = state.variables
        v = state.var

    var.pop('i', None)
    return var.copy()

def ifstatement(inputs, var, v, funcs, cond, *run):
    run = '\n'.join(run)
    if Script(cond, inputs, loop = True, vars_ = [var, v, funcs]).variables['#']:
        state = Script(run, inputs, vars_ = [var, v, funcs])
        var = state.variables
        v = state.var
    return var.copy()

def convert_version(version):
    # In  : 3
    # Out : vers/v3.py
    # In  : 4.3
    # Out : vers/v4v3.py
    # In  : 2.5.1
    # Out : vers/v2v5v1.py

    if version.endswith('.0') and version.count('.') == 1:
        version = str(int(eval(version)))
        
    version = version.split('.')
    folder = 'vers.v' + 'v'.join(version)

    numver = eval('.'.join(version[:2]))
    
    return folder, numver

def add(x,y):
    return y + x

def subtract(x,y):
    return y - x

def multiply(x,y):
    return y * x

def divide(x,y):
    return y / x

def exponent(x,y):
    return y ** x

def modulo(x,y):
    return y % x

def isprime(x):
    for i in range(2,x):
        if x%i == 0:
            return 0
    return int(x > 1 and isinstance(x, int))

def groupby(func, array):
    groups = {}
    results = list(map(func, array))
    for i, value in enumerate(array):
        if results[i] not in list(groups.keys()):
            groups[results[i]] = [value]
        else:
            groups[results[i]] += [value]
    return list(map(lambda a: a[-1], sorted(groups.items(), key = lambda a: a[0])))

def flatten_array(array):
    flat = []
    if isinstance(array, list):
        for item in array:
            flat += flatten_array(item)
    else:
        flat.append(array)
    return flat

def tighten(array):
    tight = []
    if isinstance(array, list):
        for item in array:
            if isinstance(item, list):
                tight += item
            else:
                tight.append(item)
    else:
        tight.append(array)
    return tight

def base(value, basediv):
    digits = []
    sign = (value > 0) - (value < 0)
    value = abs(value)
    while value:
        value, digit = divmod(value, basediv)
        digits.append(digit)
    return list(map(lambda v: v * sign, digits[::-1]))

def unbase(digits, base):
    total = 0
    for power, digit in enumerate(digits):
        total += digit * base ** power
    return total

def deduplicate(array):
    final = []
    for value in array:
        if value not in final:
            final.append(value)
    return final

def initiate(settings, executor):
    if settings.error:
        if settings.vernum > 5.5:
            executor(settings.code, settings.input,
                     [settings.implicit, settings.specify],
                     settings.tokens, settings.debug)
        else:
            executor(settings.code, settings.input,
                     [settings.implicit, settings.specify], settings.tokens)
    else:
        try:
            if settings.vernum > 5.5:
                executor(settings.code, settings.input,
                         [settings.implicit, settings.specify],
                         settings.tokens, settings.debug)
            else:
                executor(settings.code, settings.input,
                         [settings.implicit, settings.specify], settings.tokens)
        except Exception as err:
            print(err, file = sys.stderr)

def transvanilla(code):
    lookup = {
        'Factorial'     :   '!',
        'Negate'        :   '~',
        'AddOut'        :   '&',
        'Double'        :   '#',
        'Halve'         :   '@',
        'LogNot'        :   'N',
        'PrintNewline'  :   'P',
        'Output'        :   'O',
        'Print'         :   'H',
        'SquareRoot'    :   'S',
        'StoreInput'    :   '_',
        'Store'         :   'V',
        'Swap'          :   '}',
        'RandomNatural' :   'R',
        'Retrieve'      :   'G',
        'GlobalStore'   :   'K',
        'GlobalRetrieve':   'U',

        'Add'           :   '+',
        'Subtract'      :   '-',
        'MultiplyBy'    :   '*',
        'DivideBy'      :   '/',
        'FloorDivideBy' :   '\\',
        'RaiseToPower'  :   '^',
        'GreaterThan'   :   '>',
        'LessThan'      :   '<',
        'ModuloBy'      :   '%',
        'Equals'        :   '=',
        'RandomBetween' :   'R',

        'EachAsX'       :   'EX',
        'EachAsY'       :   'EY',
        'WhileEqual'    :   'W=',
        'WhileNotEqual' :   'W!',
        'While'         :   'W',
        'For'           :   'F',
        'If'            :   'I',
        }

    final = []
    for line in process(code.split('\n')):
        for verb in lookup:
            if line.startswith(verb):
                line = line.replace(verb, lookup[verb]).replace(' ', '')
                break
        final.append(line)
    return '\n'.join(final)

def transfunccode(code, decl = None, arg = None):
    arglookup = {
        'Arg'       :   '@',
        'Stack'     :   '*',
        'String'    :   '^',
        'Variable'  :   '?',
        'Output'    :   ':',
        'Wrap'      :   '!',
        'Unpack'    :   '~',
        }
    
    codelookup = {
        'Not'               :   '!',
        'Sort'              :   '#',
        'Swap'              :   '$',
        'Modulo'            :   '%',
        'And'               :   '&',
        'Double'            :   "'",
        'DecActive'         :   '(',
        'IncActive'         :   ')',
        'Multiply'          :   '*',
        'Add'               :   '+',
        'Divide'            :   '/',
        'LessThan'          :   '<',
        'EqualTo'           :   '=',
        'GreaterThan'       :   '>',
        'Sign'              :   '?',
        'Reverse'           :   '@',
        'Arguments'         :   'A',
        'Head'              :   'B',
        'Character'         :   'C',
        'FromIndex'         :   'D',
        'Enumerate'         :   'E',
        'Factors'           :   'F',
        'Retrieve'          :   'G',
        'StringOutput'      :   'H',
        'JoinAll'           :   'J',
        'Length'            :   'L',
        'Maximum'           :   'M',
        'JoinNewline'       :   'N',
        'Ordinal'           :   'O',
        'Prime'             :   'P',
        'Deduplicate'       :   'Q',    #
        'Range'             :   'R',
        'DeduplicateFlat'   :   'S',
        'StorePopped'       :   'V',
        'FilterUncontains'  :   'W',    #
        'RepeatSinglePop'   :   'X',
        'RandomChoice'      :   'Y',    #
        'FilterFalsey'      :   'Z',    #
        'PreviousReturn'    :   '[',
        'CallLambda'        :   ']',
        'Exponent'          :   '^',
        'Subtract'          :   '_',
        'Exponentiation'    :   '`',    #
        'ListArguments'     :   'a',
        'ToBase'            :   'b',    #
        'ClearStack'        :   'c',
        'Duplicate'         :   'd',
        'Contains'          :   'e',
        'PrimeFactors'      :   'f',
        'StackPrint'        :   'h',
        'ToInteger'         :   'i',
        'Join'              :   'j',
        'Minimum'           :   'm',
        'JoinNewlines'      :   'n',
        'Or'                :   'o',
        'Pop'               :   'p',
        'ToSet'             :   'q',
        'DyadicRange'       :   'r',
        'Sum'               :   's',
        'Split'             :   't',
        'Evaluate'          :   'v',
        'FilterContains'    :   'w',    #
        'RepeatWithoutPop'  :   'x',
        'FlatPoppingRepeat' :   'y',
        'AbsoluteValue'     :   '|',
        'BitFlip'           :   '~',    #

        'GlobalRetrieve'    :   'BK',
        'GlobalAssign'      :   'Bk',
    }

    maplookup = {

        'StackMap:'         :   'B',
        'SingleMap:'        :   'b',

        'Each:'             :   '€',
        'Sort:'             :   '§',
        'Max:'              :   '«',
        'Min:'              :   '»',
        'FilterKeep:'       :   'Þ',
        'FilterDiscard:'    :   'þ',
        'Reduce:'           :   '¦',
        'Accumulate:'       :   '¬',
        'StarMap:'          :   '£',
        'AllTrue:'          :   'ª',
        'AnyTrue:'          :   'º',
        'TakeWhile:'        :   '↑',
        'DropWhile:'        :   '↓',
        'Group:'            :   '¢',
        'Neighbours:'       :   'Ñ',
        'Reverse:'          :   'Ω',
        'Splat:'            :   'ß',
        
        'Ternary:'          :   '¿',
        ':EndTernary'       :   '¿',
        'Then:'             :   ',',
        'Else:'             :   ',',
        }
        
    if decl is arg is None:
        decl, arg, code = list(map(str.strip, split(code, '>')))
        decl = re.sub(r'^Declare ([^,]+)$', r'D,\1,', decl)
    arg = ''.join(map(arglookup.get, arg.split())) + ','

    tokens = split(code, ' ')
    for i, tkn in enumerate(tokens):
        repl = False
        
        for m in maplookup:
            if tkn.startswith(m):
                tkn = tkn.replace(m, maplookup[m])
                repl = True
                
        for t in codelookup:
            if repl:
                if tkn[1:] == t:
                    tkn = tkn.replace(t, codelookup[t])
            else:
                if tkn == t:
                    tkn = tkn.replace(t, codelookup[t])
        tokens[i] = tkn

    return decl + arg + ''.join(tokens)

def translambcode(code):
    _, arg, code = list(map(str.strip, split(code, '>')))
    return transfunccode(code, 'L', arg)

def transcallcode(code):
    repls = {
        'Call '     :   '$',
        ' With '    :   '>',
        ' And '     :   '>',
        'Input'     :   '?'
        }
    for r in repls:
        code = code.replace(r, repls[r])
    return code

def transfunc(code):
    final = []
    for line in process(code.split('\n')):
        if line.startswith('Declare'):
            line = transfunccode(line)
        elif line.startswith('Lambda'):
            line = translambcode(line)
        elif line.startswith('Call'):
            line = transcallcode(line)
        else:
            line = line
        final.append(line)
    return '\n'.join(final)

def VerboseScript(code, inputs, function, tokens, debug, vall, vvan, vfun, vout, settings):
    raise NotImplementedError()
    implicit, func = function
    
    if vall or vvan:
        code = transvanilla(code)
    if vall or vfun:
        code = transfunc(code)
        
    settings.code = code
    initiate(settings, Script)
    
    if vout:
        print('\n' + code)

class Stack(list):
    
    def push(self, *values):
        for v in values:
            if v in [True, False]:
                v = int(v)
                
            try:
                self.append(v.replace("'",'"'))
            except:
                self.append(v)
            
    def pop(self, index = -1):
        return super().pop(index)
    
    def peek(self, index = -1):
        return self[index]
    
    def swap(self):
        self[-1], self[-2] = self[-2], self[-1]

    def __str__(self):
        elements = self.copy()
        out = '['
        for element in elements:
            if hasattr(element, '__iter__') and type(element) != str:
                out += str(Stack(element)) + ' '
            else:
                out += repr(element) + ' '
        return out.strip() + ']'

class Null:
    def __init__(self, value = None):
        self.value = value

    def __repr__(self):
        return 'Null'

    def __str__(self):
        return 'Null'

    def __eq__(self, other):
        return other == Null or isinstance(other, Null)

class StackScript:

    def __call__(self, code):
        args = [
            self.name,
            code,
            [
                self.args,
                self.varargs
            ],
            self.functions,
            Stack(self.args),
            self.line,
            self.outer,
            self.tokens,
            self.vars
        ]
        
        return StackScript(*args)

    def __init__(self, name, code, args, funcs, stack,
                 line, outer, tokens, vars_, recur = False):
        
        self.args, varargs = args
        self.register = self.args.copy() if self.args else 0
        self.stacks = [stack]
        self.index = 0
        
        self.quicks = ''.join(list(self.QUICKS.keys()))
        self.prevcall = None
        self.recur = recur
        self.name = name
        
        self.functions = funcs
        self.line = line
        self.outer = outer
        self.tokens = tokens
        self.vars = vars_
        
        self.code = self.tokenize(code + ' ', tokens)

        self.varargs = {}
        while varargs:
            self.varargs.update(varargs.pop(0))
        
        cont = False

        outer = outer.split('\n')
        
        for i, cmd in enumerate(self.code):
            while True:
                try:
                    self.stack = self.stacks[self.index]
                    break
                except:
                    self.stacks.append(Stack())
                    
            if cont:
                cont -= 1
                continue
            
            if cmd[0] == '"':
                self.stack.push(cmd[1:])
                
            elif type(cmd) == list:
                quick, *cmd = cmd
                self.runquick(quick, cmd)
                
            elif cmd[0] == '{' and cmd[-1] == '}':
                
                if cmd[1] == ',':
                    argcount = abs(int(self.stack.pop()))
                    sslice = 2
                else:
                    argcount = 0
                    sslice = 1
                    
                try:
                    func = self.functions[cmd[sslice:-1]]
                except:
                    error.UnableToRetrieveFunctionError(line, outer[line-1], cmd[1:-1])
                
                feed = []
                if func.lamb:
                    feed.extend(list(self.stack))
                    self.stack.clear()
                else:
                    while len(feed) < (argcount or func.args):
                        feed.append(self.stack.pop())
                    feed = feed[::-1]

                if cmd[sslice:-1] == self.name:
                    self.prevcall = func(*feed, funccall = True, recur = True)
                else:
                    self.prevcall = func(*feed, funccall = True)
                    
                self.stack.push(self.prevcall)
                
            elif isdigit(cmd):
                
                while cmd.startswith('0'):
                    self.stack.push(0)
                    cmd = cmd[1:]
                    
                if cmd:
                    self.stack.push(eval_(cmd))

            elif cmd[0] == '`':
                var = cmd.strip('`')
                if var[0] == '$':
                    d = self.functions
                else:
                    d = self.vars
                self.stack.push(d[var])
                
            else:
                
                cmd = cmd.strip()
                
                if not cmd:
                    continue
                
                if cmd == 'Q':
                    if self.stack.pop():
                        cont = -1
                    continue
                
                try:
                    arity, command = self.COMMANDS[cmd]
                    if arity < 0:
                        arity = 0
                    result = command(*[self.stack.pop() for _ in range(arity)])
                    
                except TypeError:
                    error.IncongruentTypesError(line, outer[line-1], cmd)
                
                except:
                    error.EmptyStackError(line, outer[line-1])
                    
                if result == Null:
                    error.InvalidSymbolError(line, outer[line-1], cmd)

                if type(result) == Stack:
                    self.stack.clear()
                    self.stack.push(*result)

                elif result is not None and result != []:
                    self.stack.push(result)

    def runquick(self, quick, cmd):
        if len(cmd) == 1:
            cmd = cmd[0]

        if quick == 'Λ':
            ret = (self.varref(*cmd), 1)

        elif quick == 'Ξ':
            ret = (self.varref(''.join(cmd)), 1)

        elif cmd[0] == '{' and cmd[-1] == '}':
            cmd = ''.join(cmd)
            func = self.functions[cmd[1:-1]]
            ret = self.QUICKS[quick][1]((func.args, func), self.stack.pop())
            
        elif type(cmd) == list:
            ret = self.QUICKS[quick][1](cmd, self.stack.pop())
            
        else:
            
            cmd = ''.join(cmd).strip()
            if cmd not in self.COMMANDS or self.COMMANDS[cmd][0] == -1:
                self.runquick(quick, '{' + cmd + '}')
                return
                
            else:
                ret = self.QUICKS[quick][1](self.COMMANDS[cmd], self.stack.pop())

        ret, op = ret

        if op == 0:
            self.stacks[self.index] = ret
        if op == 1:
            self.stacks[self.index].push(ret)
        if op == 2:
            self.stacks[self.index].extend(ret)

    def tokenize(self, text, output):
        
        final = []
        
        # Num, String, Call, Var, Ref
        temps = ['', '', '', '', '']
        
        instr = False
        incall = False
        invar = False
        
        for i, char in enumerate(text):

            if char == '¡':
                if self.recur:
                    break
                else:
                    continue
            
            if char == '"': instr = not instr
            if char == '{': incall = True
            if char == '}': incall = False; temps[2] += '}'; continue
            if char == '`': invar = not invar;
	    
            if instr:    temps[1] += char
            elif incall: temps[2] += char
            elif invar:  temps[3] += char
            else:
                if temps[1]:
                    final.append(temps[1])
                    temps[1] = ''
                    
                if temps[2]:
                    final.append(temps[2])
                    temps[2] = ''
                    
                if temps[3]:
                    final.append(temps[3])
                    temps[3] = ''
                    
                if isdigit(char):
                    try:
                        if char == '-':
                            if text[text.index(char)+1].isdigit():
                                temps[0] += char
                        else:
                            temps[0] += char
                    except: final.append(char)
                    
                else:
                    if temps[0]:
                        final.append(temps[0])
                        temps[0] = ''
                    final.append(char)

        if temps[0]: final.append(temps[0])
        if temps[1]: final.append(temps[1])
        if temps[2]: final.append(temps[2])
        if temps[3]: final.append(temps[3])
        del temps
        
        tokens = []
        for i, f in enumerate(final):
            if f in 'Bb' and final[i+1] not in self.quicks + '{':
                tokens.append(f + final.pop(i + 1))
            elif f in '" `':
                pass
            else:
                tokens.append(f)

        chain = []

        index = 0
        while index < len(tokens):
            if tokens[index] in self.quicks:
                hungry = self.QUICKS[tokens[index]][0]
                quick = tokens[index]
                
                chain.append([quick])

                index += 1

                if hungry:
                    for i in range(2):
                        if tokens[index] in self.quicks:
                            chain[-1] += tokens[index]
                            index += 1
                        
                    chain[-1] += tokens[index]
                    index += 1
                    
                else:
                    while tokens[index] != quick:
                        chain[-1] += tokens[index]
                        index += 1
                    index += 1
            else:
                chain.append(tokens[index])
                index += 1

        chain = list(filter(None, chain))

        for index, element in enumerate(chain):
            if isinstance(element, list):
                tkns = []
                last = False
                for char in element:
                    if last == 2:
                        tkns[-1] += char
                        last = False
                        continue
                        
                    if char == '{':
                        tkns.append('')
                        last = True

                    if char == '}':
                        tkns[-1] += '}'
                        last = False
                        continue

                    if char in 'Bb' and not last:
                        tkns.append(char)
                        last = 2
                        continue
                        
                    if last:
                        tkns[-1] += char
                    else:
                        tkns.append(char)
                chain[index] = tkns

        if output:
            print(chain)
        return chain

    @property
    def QUICKS(self):
        # ⁇⁈⁉ΣΠ
        return {
                '€': ( 1, self.quickeach                                ),
                '§': ( 1, self.quicksort                                ),
                '«': ( 1, self.quickmax                                 ),
                '»': ( 1, self.quickmin                                 ),
                'Þ': ( 1, self.quickfiltertrue                          ),
                'þ': ( 1, self.quickfilterfalse                         ),
                '¦': ( 1, self.quickreduce                              ),
                '¬': ( 1, self.quickaccumulate                          ),
                '£': ( 1, self.quickstareach                            ),
                'ª': ( 1, self.quickall                                 ),
                'º': ( 1, self.quickany                                 ),
                '↑': ( 1, self.quicktakewhile                           ),
                '↓': ( 1, self.quickdropwhile                           ),
                '¢': ( 1, self.quickgroupby                             ),
                'Ñ': ( 1, self.quickneighbours                          ),
                'Ω': ( 1, self.quickreverse                             ),
                '¿': ( 0, self.quickternary                             ),
                'ß': ( 1, self.quicksplat                               ),
                'Δ': ( 1, self.quickgroupadjacent                       ),
                '×': ( 1, self.quickunwrap                              ),
                '‽': ( 1, self.quicktable                               ),
                
                'Λ': ( 1, self.varref                                   ),
                'Ξ': ( 0, self.varref                                   ),
               }
    
    @property
    def COMMANDS(self):
        return {
                '!': ( 1, lambda x: int(not x)                          ),
                '#': ( 0, lambda: self.stack.sort()                     ),
                '$': ( 0, lambda: self.stack.swap()                     ),
                '%': ( 2, lambda x, y: modulo(x, y)                     ),
                '&': ( 2, lambda x, y: x and y                          ),
                "'": ( 1, lambda x: x * 2                               ),
                '(': ( 0, lambda: self.decrement()                      ),
                ')': ( 0, lambda: self.increment()                      ),
                '*': ( 2, lambda x, y: multiply(x, y)                   ),
                '+': ( 2, lambda x, y: add(x, y)                        ),
                '/': ( 2, lambda x, y: divide(x, y)                     ),
                ':': ( 2, lambda x, y: x[y]                             ),
                '<': ( 2, lambda x, y: int(x < y)                       ),
                '=': ( 2, lambda x, y: int(x == y)                      ),
                '>': ( 2, lambda x, y: int(x > y)                       ),
                '?': ( 1, lambda x: (x > 0) - (x < 0)                   ),
                '@': ( 0, lambda: self.stack.reverse()                  ),
                
                'A': ( 0, lambda: self.stack.push(*self.args)           ),
                'B': ( 1, lambda x: self.stack[:x]                      ),
                'C': ( 1, lambda x: chr(x)                              ),
                'D': ( 1, lambda x: self.stack[-x]                      ),
                'E': ( 1, lambda x: list(map(list, enumerate(x, 1)))    ),
                'F': ( 1, lambda x: self.stack.push(*self.factors(x))   ),
                'G': ( 0, lambda: self.stack.push(self.register)        ),
                'H': ( 0, lambda: print(''.join(map(str, self.stack)))  ),
                'I': (-1, lambda: Null                                  ),
                'J': ( 0, lambda: self.join('')                         ),
                'K': (-1, lambda: Null                                  ),
                'L': ( 0, lambda: len(self.stack)                       ),
                'M': ( 0, lambda: max(self.stack)                       ),
                'N': ( 0, lambda: '\n'.join(map(str, self.stack))       ),
                'O': ( 1, lambda x: ord(x)                              ),
                'P': ( 1, lambda x: isprime(x)                          ),
                'R': ( 1, lambda x: list(range(1, x+1))                 ),
                'S': ( 0, lambda: self.remove_duplicates()              ),
                'T': ( 2, lambda x, y: [x[z: z+y] for z in range(0, len(x) - y + 1, y)] + ([x[len(x) - y + 1:]] if len(x) % y else [])	),
                'U': (-1, lambda: Null                                  ),
                'V': ( 1, lambda x: self.store(x)                       ),
                'X': ( 2, lambda x, y: [x for _ in range(y)]            ),
                'Y': (-1, lambda: Null                                  ),
                'Z': (-1, lambda: Null                                  ),

                '[': ( 0, lambda: self.prevcall                         ),
                ']': ( 1, lambda x: self.run_lambda(x)                  ),
                '^': ( 2, lambda x, y: exponent(x, y)                   ),
                '_': ( 2, lambda x, y: subtract(x, y)                   ),
                '`': (-1, lambda: Null                                  ),

                'a': ( 0, lambda: list(self.args)                       ),
                'b': (-1, lambda: Null                                  ),
                'c': ( 0, lambda: self.stack.clear()                    ),
                'd': ( 0, lambda: self.stack.push(self.stack[-1])       ),
                'e': ( 2, lambda x, y: x in y                           ),
                'f': ( 0, lambda: list(filter(isprime, self.factors())) ),
                'g': (-1, lambda: Null                                  ),
                'h': ( 0, lambda: print(self.stack)                     ),
                'i': ( 1, lambda x: int(x)                              ),
                'j': ( 1, lambda x: self.join(str(x))                   ),
                'k': (-1, lambda: Null                                  ),
                'l': (-1, lambda: Null                                  ),
                'm': ( 0, lambda: min(self.stack)                       ),
                'n': ( 0, lambda: self.join()                           ),
                'o': ( 2, lambda x, y: x or y                           ),
                'p': ( 1, lambda x: None                                ),
                'q': ( 1, lambda x: deduplicate(x)                      ),
                'r': ( 2, lambda x, y: list(range(x, y+1))              ),
                's': ( 0, lambda: sum(self.stack)                       ),
                't': ( 2, lambda x, y: str(x).split(str(y))             ),
                'u': (-1, lambda: Null                                  ),
                'v': ( 1, lambda x: eval(x)                             ),
                'w': (-1, lambda: Null                                  ),
		'x': ( 1, lambda x: [self.stack[-1] for _ in range(x)]  ),
		'y': ( 1, lambda x: [self.stack.push(self.stack[-1]) for _ in range(x)][:0]         ),
                'z': ( 2, lambda x, y: list(map(list, zip(x, y)))       ),
                
                '|': ( 1, lambda x: abs(x)                              ),
                '~': ( 2, lambda f, x: f(*x) if hasattr(x, '__iter__') else f(x)                    ),

                'B!':( 0, lambda: self.a(lambda a: int(not a))          ),
                'B#':( 0, lambda: self.a(sorted)                        ),
                'B$':(-1, lambda: Null                                  ),
                'B%':( 0, lambda: self.a(lambda l: fn.reduce(op.mod, l))),
                'B&':( 0, lambda: self.a(lambda l: fn.reduce(op.and_, l))                           ),
                "B'":( 0, lambda: self.a(lambda x: 2 * x)               ),
                'B(':( 0, lambda: self.stacks[self.index - 1].pop()     ),
                'B)':( 0, lambda: self.stacks[(self.index + 1) % len(self.stacks)].pop()            ),
                'B*':( 0, lambda: self.a(lambda l: fn.reduce(op.mul, l))),
                'B+':( 0, lambda: self.a(lambda l: fn.reduce(op.add, l))),
                'B/':( 0, lambda: self.a(lambda l: fn.reduce(op.truediv, l))                        ),
                'B:':(-1, lambda: Null                                  ),
                'B<':(-1, lambda: Null                                  ),
                'B=':( 0, lambda: self.a(lambda l: self.eq(*l))         ),
                'B>':(-1, lambda: Null                                  ),
                'B?':(-1, lambda: Null                                  ),
                'B@':( 0, lambda: self.a(reversed, True)                ),
                
                'BA':( 0, lambda: self.a(abs)                           ),
                'BB':( 1, lambda x: base(x, 2)                          ),
                'BC':( 0, lambda: self.collect()                        ),
                'BD':( 0, lambda: self.a(lambda i: list(map(int, str(i))))                          ),
                'BE':( 0, lambda: self.a(lambda i: int(i in self.stack[-1]))                        ),
                'BF':( 0, lambda: self.flatten()                        ),
                'BG':( 1, lambda x: [list(g) for k, g in it.groupby(x)] ),
                'BH':(-1, lambda: Null                                  ),
                'BI':(-1, lambda: Null                                  ),
                'BJ':( 0, lambda: self.a(lambda i: ''.join(map(str, i)))),
                'BK':( 0, lambda: GLOBALREGISTER                        ),
                'BL':( 0, lambda: self.a(len)                           ),
                'BM':( 0, lambda: self.a(max)                           ),
                'BN':(-1, lambda: Null                                  ),
                'BO':(-1, lambda: Null                                  ),
                'BP':( 0, lambda: self.a(lambda x: x[1:])               ),
                'BQ':( 0, lambda: self.a(self.remove_duplicates)        ),
                'BR':( 0, lambda: self.a(lambda x: list(range(1, x + 1)))                           ),
                'BS':( 0, lambda: Stack([self.stack[i : i+2] for i in range(len(self.stack) - 1)])  ),
                'BT':(-1, lambda: Null                                  ),
                'BU':(-1, lambda: Null                                  ),
                'BV':( 1, lambda x: exec(x)                             ),
                'BW':( 0, lambda: Stack([i for i in self.stack[:-1] if i not in self.stack[-1]])    ),
                'BX':( 1, lambda x: random.choice(x)                    ),
                'BY':( 0, lambda: self.a(random.choice)                 ),
                'BZ':( 0, lambda: Stack(filter(None, self.stack))       ),

                'B]':( 0, lambda: self.wrap()                           ),
                'B[':( 0, lambda: self.a(lambda l: [l])                 ),
                'B^':( 0, lambda: self.a(lambda l: fn.reduce(op.xor, l))),
                'B_':( 0, lambda: self.a(lambda l: fn.reduce(op.sub, l))),
                'B`':( 0, lambda: self.a(lambda l: fn.reduce(op.pow, l))),

                'Ba':( 2, lambda x, y: x & y                            ),
                'Bb':( 2, lambda x, y: unbase(x, y)                     ),
                'Bc':( 0, lambda: self.columns()                        ),
                'Bd':( 0, lambda: self.a(lambda l: fn.reduce(op.floordiv, l))                       ),
                'Be':( 1, lambda x: [int(i in self.stack[-1])for i in x]),
                'Bf':( 1, lambda x: ~x                                  ),
                'Bg':(-1, lambda: Null                                  ),
                'Bh':( 1, lambda x: print(x)                            ),
                'Bi':( 0, lambda: self.a(int)                           ),
                'Bj':( 0, lambda: self.a(isprime)                       ),
                'Bk':( 1, lambda x: self.assign(x)                      ),
                'Bl':(-1, lambda: Null                                  ),
                'Bm':( 0, lambda: self.a(min)                           ),
                'Bn':( 0, lambda: self.a(lambda i: -i)                  ),
                'Bo':( 2, lambda x, y: x | y                            ),
                'Bp':( 0, lambda: self.a(lambda x: x[:-1])              ),
                'Bq':(-1, lambda: Null                                  ),
                'Br':(-1, lambda: Null                                  ),
                'Bs':( 0, lambda: self.a(sum)                           ),
                'Bt':(-1, lambda: Null                                  ),
                'Bu':(-1, lambda: Null                                  ),
                'Bv':( 0, lambda: self.a(lambda i: int(''.join(map(str, i))))                       ),
                'Bw':( 0, lambda: Stack([i for i in self.stack[:-1] if i in self.stack[-1]])        ),
                'Bx':( 2, lambda x, y: x ^ y                            ),
                'By':(-1, lambda: Null                                  ),
                'Bz':(-1, lambda: Null                                  ),
                
                'B|':( 0, lambda: self.a(lambda l: fn.reduce(op.or_, l)) ),
                'B~':( 0, lambda: self.a(op.inv)                         ),
                
                'b!':( 1, lambda x: list(map(lambda a: int(not a), x))   ),
                'b#':(-1, lambda: Null                                   ),
                'b$':(-1, lambda: Null                                   ),
                'b%':( 1, lambda x: fn.reduce(op.mod, x)                 ),
                'b&':( 1, lambda x: fn.reduce(op.and_, x)                ),
                "b'":( 1, lambda x: [i * 2 for i in x]                   ),
                'b(':( 1, lambda x: self.stacks[self.index - 1].push(x)  ),
                'b)':( 1, lambda x: self.stacks[(self.index + 1) % len(self.stacks)].push(x)         ),
                'b*':( 1, lambda x: fn.reduce(op.mul, x)                 ),
                'b+':( 1, lambda x: fn.reduce(op.add, x)                 ),
                'b/':( 1, lambda x: fn.reduce(op.truediv, x)             ),
                'b:':(-1, lambda: Null                                   ),
                'b<':( 2, lambda x, y: min(x, y)                         ),
                'b=':( 1, lambda x: self.eq(*x)                          ),
                'b>':( 2, lambda x, y: max(x, y)                         ),
                'b?':(-1, lambda: Null                                   ),
                'b@':(-1, lambda: Null                                   ),

                'bA':(-1, lambda: Null                                   ),
                'bB':( 0, lambda: self.pad_bin()                         ),
                'bC':(-1, lambda: Null                                   ),
                'bD':(-1, lambda: Null                                   ),
                'bE':(-1, lambda: Null                                   ),
                'bF':( 1, lambda x: flatten_array(x)                     ),
                'bG':(-1, lambda: Null                                   ),
                'bH':(-1, lambda: Null                                   ),
                'bI':(-1, lambda: Null                                   ),
                'bJ':(-1, lambda: Null                                   ),
                'bK':(-1, lambda: Null                                   ),
                'bL':( 1, lambda x: len(x)                               ),
                'bM':( 1, lambda x: max(x)                               ),
                'bN':(-1, lambda: Null                                   ),
                'bO':(-1, lambda: Null                                   ),
                'bP':(-1, lambda: Null                                   ),
                'bQ':(-1, lambda: Null                                   ),
                'bR':( 1, lambda x: x[::-1]                              ),
                'bS':(-1, lambda: Null                                   ),
                'bT':(-1, lambda: Null                                   ),
                'bU':( 1, lambda x: self.stack.push(*x)                  ),
                'bV':( 1, lambda x: self.selfexec(x)                     ),
                'bW':(-1, lambda: Null                                   ),
                'bX':(-1, lambda: Null                                   ),
                'bY':(-1, lambda: Null                                   ),
                'bZ':(-1, lambda: Null                                   ),

                'b[':( 2, lambda x, y: [x, y]                            ),
                'b]':( 1, lambda x: [x]                                  ),
                'b^':( 1, lambda x: fn.reduce(op.xor, x)                 ),
                'b_':( 1, lambda x: fn.reduce(op.sub, x)                 ),
                'b`':( 1, lambda x: fn.reduce(op.pow, x)                 ),

                'ba':(-1, lambda: Null                                   ),
                'bb':( 2, lambda x, y: base(x, y)                        ),
                'bc':(-1, lambda: Null                                   ),
                'bd':( 1, lambda x: fn.reduce(op.floordiv, x)            ),
                'be':(-1, lambda: Null                                   ),
                'bf':(-1, lambda: Null                                   ),
                'bg':(-1, lambda: Null                                   ),
                'bh':(-1, lambda: Null                                   ),
                'bi':( 1, lambda x: x                                    ),
                'bj':(-1, lambda: Null                                   ),
                'bk':(-1, lambda: Null                                   ),
                'bl':(-1, lambda: Null                                   ),
                'bm':( 1, lambda x: min(x)                               ),
                'bn':(-1, lambda: Null                                   ),
                'bo':(-1, lambda: Null                                   ),
                'bp':(-1, lambda: Null                                   ),
                'bq':(-1, lambda: Null                                   ),
                'br':(-1, lambda: Null                                   ),
                'bs':(-1, lambda: Null                                   ),
                'bt':(-1, lambda: Null                                   ),
                'bu':(-1, lambda: Null                                   ),
                'bv':(-1, lambda: Null                                   ),
                'bw':(-1, lambda: Null                                   ),
                'bx':(-1, lambda: Null                                   ),
                'by':(-1, lambda: Null                                   ),
                'bz':(-1, lambda: Null                                   ),
                
                'b|':( 1, lambda x: fn.reduce(op.or_, x)                 ),
                'b~':( 1, lambda x: list(map(op.inv, x))                 ),
               }

    def a(self, func, array = False):
        if array:
            self.stacks[self.index] = Stack(map(lambda v: list(func(v)), self.stack))
        self.stacks[self.index] = Stack(map(func, self.stack))

    def assign(self, value):
        global GLOBALREGISTER
        GLOBALREGISTER = value
        
    def collect(self, iterable = None):
        usestack = False
        if iterable is None:
            iterable = self.stack
            usestack = True
            
        array = []
        sub_array = []
        for element in iterable:
            if type(element) == list:
                if sub_array:
                    array.append(sub_array)
                    sub_array = []
                array.append(element)
            else:
                sub_array.append(element)
        if sub_array:
            array.append(sub_array)
            
        if usestack:
            self.stacks[self.index] = Stack(array)
        else:
            return array
		
    def columns(self):
        self.stacks[self.index] = Stack(map(list, zip(*self.stack)))

    def decrement(self):
        self.index -= 1
        
    def eq(self, *args):
        incs = [args[i] == args[i-1] for i in range(1, len(args))]
        return int(all(incs))
    
    def factors(self, x):
        lof = []
        if hasattr(x, '__iter__'):
            return list(x)
        
        for i in range(1, int(x)):
            if x%i == 0:
                lof.append(i)
                
        return lof
	
    def flatten(self):
        copy = list(self.stack)
        flat = flatten_array(copy)
        self.stack.clear()
        self.stack.push(*flat)

    def increment(self):
        self.index += 1
        
    def join(self, char='\n'):
        newstack = Stack()
        newstack.push(char.join(map(str, self.stack)))
        self.stacks[self.index] = newstack
		
    def pad_bin(self):
        copy = self.stack.copy()
        length = max(map(lambda a: len(bin(a)[2:]), copy))
        for i in range(len(self.stack)):
            self.stacks[self.index][i] = Stack(map(eval_, bin(self.stack[i])[2:].rjust(length, '0')))

    # Quick commands #

    def quickaccumulate(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            ret = list(it.accumulate(left, lambda x, y: quick(cmd, x, y)[0]))
        else:
            _, cmd = cmd
            ret = list(it.accumulate(left, cmd))
        return ret, 1

    def quickall(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            if arity == 1:
                ret = all(map(lambda a: quick(cmd, a), left))
            else:
                if right is None:
                    right = self.stack.pop()
                ret = all(map(lambda a: quick(cmd, a, right), left))
        else:
            arity, cmd = cmd
            if arity == 1:
                ret = all(map(cmd, left))
            else:
                if right is None:
                    right = self.stack.pop()
                ret = all(map(cmd, it.repeat(right), left))

        return ret, 1

    def quickany(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            if arity == 1:
                ret = any(map(lambda a: quick(cmd, a), left))
            else:
                if right is None:
                    right = self.stack.pop()
                ret = any(map(lambda a: quick(cmd, a, right), left))
        else:
            arity, cmd = cmd
            if arity == 1:
                ret = any(map(cmd, left))
            else:
                if right is None:
                    right = self.stack.pop()
                ret = any(map(cmd, it.repeat(right), left))

        return ret, 1

    def quickdropwhile(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            if arity == 1:
                ret = list(it.dropwhile(lambda a: quick(cmd, a), left))
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = list(it.dropwhile(lambda a: quick(cmd, a, right), left))
        else:
            arity, cmd = cmd
            if arity == 2:
                if right is None:
                    right = self.stack.pop()
                ret = list(it.dropwhile(lambda a: cmd(a, right), left))
            else:
                ret = list(it.dropwhile(cmd, left))

        return ret, 1

    def quickeach(self, cmd, left, right = None):
        if isinstance(cmd, list):
            print('€', cmd)
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            if arity == 1:
                ret = list(map(lambda a: quick(cmd, a)[0], left))
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = list(map(lambda a: quick(cmd, a, right)[0], left))
        else:
            arity, cmd = cmd
            if arity == 1:
                ret = list(map(cmd, left))
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = list(map(lambda a: cmd(a, right), left))

        return ret, 1

    def quickfilterfalse(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            if arity == 1:
                ret = list(it.filterfalse(lambda a: quick(cmd, a), left))
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = list(it.filterfalse(lambda a: quick(cmd, a, right), left))
        else:
            arity, cmd = cmd
            if arity == 2:
                if right is None:
                    right = self.stack.pop()
                ret = list(it.filterfalse(lambda a: cmd(a, right), left))
            else:
                ret = list(it.filterfalse(cmd, left))

        return ret, 1

    def quickfiltertrue(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            if arity == 1:
                ret = list(filter(lambda a: quick(cmd, a), left))
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = list(filter(lambda a: quick(cmd, a, right), left))
        else:
            arity, cmd = cmd
            if arity == 2:
                if right is None:
                    right = self.stack.pop()
                ret = list(filter(lambda a: cmd(a, right), left))
            else:
                ret = list(filter(cmd, left))

        return ret, 1

    def quickgroupadjacent(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            if arity == 1:
                ret = it.groupby(left, lambda a: quick(cmd, a))
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = it.groupby(left, lambda a: quick(cmd, a, right))
        else:
            arity, cmd = cmd
            if arity == 2:
                if right is None:
                    right = self.stack.pop()
                ret = it.groupby(left, lambda a: cmd(a, right))
            else:
                ret = it.groupby(left, cmd)

        ret = list(map(lambda a: list(a[1]), ret))

        return ret, 1

    def quickgroupby(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            if arity == 1:
                ret = list(groupby(lambda a: quick(cmd, a), left))
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = list(groupby(lambda a: quick(cmd, a, right), left))
        else:
            arity, cmd = cmd
            if arity == 2:
                if right is None:
                    right = self.stack.pop()
                ret = list(groupby(lambda a: cmd(a, right), left))
            else:
                ret = list(groupby(cmd, left))

        return ret, 1

    def quickmax(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            if arity == 1:
                ret = max(left, key = lambda a: quick(cmd, a))
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = max(left, key = lambda a: quick(cmd, a, right))
        else:
            arity, cmd = cmd
            if arity == 1:
                ret = max(left, key = cmd)
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = max(left, key = lambda a: cmd(a, right))

        return ret, 1

    def quickmin(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            if arity == 1:
                ret = min(left, key = lambda a: quick(cmd, a))
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = min(left, key = lambda a: quick(cmd, a, right))
        else:
            arity, cmd = cmd
            if arity == 1:
                ret = min(left, key = cmd)
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = min(left, key = lambda a: cmd(a, right))

        return ret, 1

    def quickneighbours(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            pairs = [left[i: i+2] for i in range(len(left) - 1)]
            if arity == 2:
                ret = [quick(cmd, *sub) for sub in pairs]
            else:
                ret = [quick(cmd, sub) for sub in pairs]
        else:
            arity, cmd = cmd
            pairs = [left[i: i+2] for i in range(len(left) - 1)]
            if arity == 2:
                ret = [cmd(*sub) for sub in pairs]
            else:
                ret = [cmd(sub) for sub in pairs]

        return ret, 1

    def quickreduce(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            ret = fn.reduce(lambda x, y: quick(cmd, x, y)[0], left)
        else:
            _, cmd = cmd
            ret = fn.reduce(cmd, left)
        return ret, 1
            
    def quickreverse(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            if arity == 1:
                ret = quick(cmd, left)[0]
            else:
                if right is None:
                    right = self.stack.pop()
                ret = quick(cmd, right, left)[0]
        else:
            arity, cmd = cmd
            if arity == 1:
                ret = cmd(left)
            else:
                if right is None:
                    right = self.stack.pop()
                ret = cmd(right, left)
                
        return ret, 1

    def quicksort(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            if arity == 1:
                ret = sorted(left, key = lambda a: quick(cmd, a))
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = sorted(left, key = lambda a: quick(cmd, a, right))
        else:
            arity, cmd = cmd
            if arity == 1:
                ret = sorted(left, key = cmd)
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = sorted(left, key = lambda a: cmd(a, right))

        return ret, 1

    def quicksplat(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            ret = quick(cmd, left, right)[0]
        else:
            arity, cmd = cmd

            if arity == 2:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = list(map(lambda a: cmd(a, right), left))
            elif arity == 1:
                ret = cmd(left)
            else:
                ret = left
                
        return ret, 2

    def quickstareach(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            ret = list(it.starmap(lambda a: quick(cmd, a)[0], left))
        else:
            _, cmd = cmd
            ret = list(it.starmap(cmd, left))

        return ret, 1

    def quicktable(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]

            pairs = []
            if right is None:
                right = self.stack.pop()
                
            ret = [quick(cmd, l, r) for l in left for r in right]
        else:
            _, cmd = cmd
            if right is None:
                right = self.stack.pop()

            ret = [cmd(l, r) for l in left for r in right]

        return ret, 1
                    
    def quicktakewhile(self, cmd, left, right = None):
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            
            try:
                arity
            except:
                arity = cmd[0]
                
            if arity == 1:
                ret = list(it.takewhile(lambda a: quick(cmd, a), left))
            else:
                if right is None:
                    right = self.stack.pop()
                left, right = right, left
                ret = list(it.takewhile(lambda a: quick(cmd, a, right), left))
        else:
            arity, cmd = cmd
            if arity == 2:
                if right is None:
                    right = self.stack.pop()
                ret = list(it.takewhile(lambda a: cmd(a, right), left))
            else:
                ret = list(it.takewhile(cmd, left))

        return ret, 1

    def quickternary(self, cmd, *_):
        cond, ifclause, elseclause = split(''.join(cmd), ',')
        if self(cond).stacks.pop().pop():
            ret = self(ifclause).stacks.pop().pop()
        else:
            ret = self(elseclause).stacks.pop().pop()
        return ret, 1

    def quickunwrap(self, cmd, left, right = None):
        if arity == 2:
            self.stack.push(right)
        self.stack.push(left)

        left = [self.stack.copy()]
        right = None
        
        if isinstance(cmd, list):
            quick, cmd = cmd
            if quick in '¬¦':
                arity = 1
            quick = self.QUICKS[quick][1]
            cmd = self.COMMANDS[cmd]
            ret = quick(cmd, left, right)[0]
        else:
            arity, cmd = cmd

            if arity == 2:
                left, right = right, left
                ret = list(map(lambda a: cmd(a, left), right))
            elif arity == 1:
                ret = cmd(left)
            else:
                ret = left
                
        return ret, 2

    # End quick commands #

    def remove(self, even_odd):
        self.stacks[self.index] = Stack(filter(lambda x: x%2 == int(bool(even_odd)), self.stack))
        
    def remove_duplicates(self, array=None):
        final = []
        if array is None: array = self.stack
        for s in array:
            if s not in final:
                final.append(s)
        return final
	
    def run(self, flag, text):
        ret = self.stacks[self.index]
        if flag:
            return ret
        if text:
            return ''.join(list(map(StackScript.stringify, ret)))
        
        try:
            final = ret.pop()
        except:
            final = 0
            
        if type(final) == list:
            return Stack(final)
        return final

    def run_lambda(self, index):
        lamb = self.functions['lambda {}'.format(index)]
        self.prevcall = lamb(*self.stack)
        self.stack.clear()
        self.stack.push(self.prevcall)

    def selfexec(self, code):
        if type(code) == int:
            code = chr(code)
        if type(code) == list:
            code = ''.join(map(str, code))
        print(code)
        
    def store(self, value):
        self.register = value
        
    @staticmethod
    def stringify(value):
        try:
            return chr(int(abs(value)))
        except:
            return str(value)

    def varref(self, var):
        if var in self.varargs.keys():
            return self.varargs[var]

        raise error.UnknownVariableError(self.line, self.outer.split('\n')[self.line-1], var)
		
    def wrap(self):
        array = self.stack.copy()
        self.stack.clear()
        self.stack.push(array)

class Function:

    def __init__(self, name, code, switches,
                 line = 0, g_code = "", outerf = {},
                 tkns = False, vars_ = {}):
        
        self.name = name
        self.code = code
        self.stack = Stack()
        self.switches = switches.copy()
        self.line = line
        self.gen = g_code
        self.outerf = outerf
        self.tkns = tkns
        self.vars = vars_
        
        self.lamb = self.switches['lambda']
        self.original = self.switches['@'].copy()

        self.calls = 0
        self.args = 0

        for elem in self.switches['@']:
            if elem == Null:
                self.args += 1
            if isinstance(elem, dict):
                self.args += sum(i == Null for i in elem.values())

    def __call__(self, *args, funccall = False, recur = False):
        prov = []
        if self.calls:
            self.switches['@'] = self.original.copy()

        self.calls += 1
        args = list(args)
        nulls = 0

        for elem in self.switches['@']:
            if elem == Null:
                nulls += 1
            if isinstance(elem, dict):
                nulls += sum(i == Null for i in elem.values())

        while len(args) < nulls:
            args.append(-1)

        filtrate = []
        for index, elem in enumerate(self.switches['@']):
            if isinstance(elem, dict):
                for key in elem.keys():
                    if elem[key] == Null:
                        elem[key] = args.pop(0)
                        
            if elem == Null:
                self.switches['@'][index] = args.pop(0)

            elem = self.switches['@'][index]
            if not isinstance(elem, dict):
                prov.append(elem)
            else:
                filtrate.append(elem)

        if self.switches['?'] or self.switches['lambda']:
            prov.extend(args)

        if self.switches['!']:
            prov = [prov[:]]

        if self.switches['~']:
            prov = tighten(prov)

        if self.switches['#']:
            prov = prov[::-1]

        self.stack.push(*prov)

        script_obj = fn.partial(

            StackScript,

            self.name,
            self.code,
            [
                prov,
                filtrate,
            ],
            self.outerf,
            self.stack,
            self.line,
            self.gen,
            self.tkns,
            self.vars,
            recur,

        )

        ret = script_obj().run(self.switches['*'], self.switches['^'])

        if isinstance(ret, (list, Stack)):
            ret = Stack(map(lambda a: int(a) if isinstance(a, bool) else a, ret))
            
        if isinstance(ret, bool):
            ret = int(ret)

        self.stack = Stack()
        self.switches['@'] = list()
        
        if self.switches[':']:
            print(ret)
            return ret
            if funccall:
                return ret
            else:
                return Null()

        return ret
        
    def __repr__(self):
        return "<Function ${}: '{}'>".format(self.name, self.code)

class Script:

    def __init__(self, code, inputs, implicit = [False, False],
                 tokens = False, debug = False, loop = False,
                 vars_ = None, history = None):

        code = code.split('\n')

        self.code = ['']
        self.preserve = inputs
        self.input = iter(inputs)
        self.implicit, self.specify = implicit
        self.display_tokens = debug
        self.func_tokens = tokens
        self.called = False
        self.history = history
        
        self.lambdas = 0

        if vars_ is None:
            self.variables = collections.OrderedDict()
            self.variables['x'] = 0; self.variables['y'] = 0
            self.var = 'x'
            self.functions = collections.OrderedDict()
            
        else:
            self.variables = vars_[0].copy()
            self.var = vars_[1]
            self.functions = vars_[2].copy()

        if loop:
            self.variables['#'] = 0
            self.var = '#'

        for index, line in enumerate(code):
            if comment.search(line):
                line = line.split(';')[0].strip()
                self.code[-1] += line
                
            else:
                line = line.split(';')[0].strip()
                self.code.append(line)

        self.code = list(filter(None, self.code))

        for index, line in enumerate(self.code, 1):
            self.index = index
            self.line = line
            
            self.fn_shell = fn.partial(Function,
                                           line     = self.index,
                                           g_code   = '\n'.join(self.code),
                                           outerf   = self.functions,
                                           tkns     = self.func_tokens,
                                           vars_    = self.variables,
                                       )

            if line == ')history':
                print(self.history)
                continue
            
            category = self.categorise(line)
            ret = category(line)

            if hasattr(ret, 'with_traceback') or isinstance(ret, fn.partial):
                raise ret(index, line)

            if self.display_tokens:
                msg = '    {:%s} {:40} {:20} {}' % max(map(len, self.code))
                print(msg.format(line, str(self.variables), str(ret), self.var))

        if self.implicit and not self.called and self.functions:

            if self.specify:
                func = self.functions[self.specify]
            else:
                func = self.functions[list(self.functions.keys())[0]]

            self.input = list(self.input)
            if self.input:
                ret = func(*self.input)
                
            elif len(self.variables.keys()) > 2:
                vals = filter(None, self.variables.values())
                ret = func(*vals)

            else:
                args = []
                if self.variables['x']:
                    args.append(self.variables['x'])
                if self.variables['y']:
                    args.append(self.variables['y'])

                ret = func(*args)

            if type(ret) != Null:
                print(ret)
                
    def categorise(self, string):
        if additionals.match(string):
            return self.additionals
        
        if function.match(string):
            return self.function
        
        if varassign.match(string):
            return self.assign
        
        if varswap.match(string):
            return self.swap
        
        if varactive.match(string):
            return self.activate

        if construct.match(string):
            return self.construct

        if infix.match(string):
            return self.infix

        if prefix.match(string):
            return self.prefix

        return lambda _: error.InvalidSyntaxError(self.index, self.line)

    def additionals(self, string):
        mode, *string = string
        cmd = ''.join(string)

        if mode == '}':
            exec(cmd)

        if mode == ']':
            arity, command = extensions.getcmd[cmd]

            if arity == 0:
                self.variables[self.var] = command()
                
            elif arity == 1:
                try:
                    self.variables[self.var] = command(self.variables[self.var])
                except Exception as exc:
                    return fn.partial(error.PythonError, error = exc)
                
            else:
                vals = list(self.variables.values())[:arity]
                try:
                    self.variables[self.var] = command(*vals)
                except Exception as exc:
                    return fn.partial(error.PythonError, error = exc)

    def assign(self, string):
        var, val = varassign.search(string).groups()
        val = self.eval(val)
        
        if hasattr(val, 'with_traceback') or isinstance(val, fn.partial):
            return val

        self.variables[var] = val

    def swap(self, string):
        left, right = varswap.search(string).groups()
        v = self.variables.copy()
        v[left], v[right] = v[right], v[left]
        self.variables = v.copy()

    def activate(self, string):
        var = varactive.search(string).groups()[0]
        self.var = var

    def construct(self, string):
        head, *captures = construct.search(string).groups()
        loop, *head = head

        cond = ''.join(head)
        
        if cond in self.variables.keys() and loop in 'WID':
            cond = 'B' + cond
            
        body = captures.pop().strip(',')
        loop = self.cons[loop]

        cmds = splitting(body, ',')

        self.variables = loop(self.input, self.variables.copy(), self.var,
                              self.functions.copy(), cond, *cmds)

    def infix(self, string):
        left, oper, right = infix.search(string).groups()

        if left == '':
            left = self.var
        if right == '':
            right = self.var
     
        if oper in self.commands.keys():
            oper = self.commands[oper]
        else:
            return fn.partial(error.InvalidSymbolError, char = oper)

        left, right = self.eval(left), self.eval(right)
        
        if hasattr(left, 'with_traceback') or isinstance(left, fn.partial):
            return left
        if hasattr(right, 'with_traceback') or isinstance(right, fn.partial):
            return right
        
        ret = oper(left, right)
        if ret is not None:
            self.variables[self.var] = ret

    def prefix(self, string):
        oper, arg = prefix.search(string).groups()

        if arg == '':
            arg = self.var

        if oper in self.commands.keys():
            oper = self.commands[oper]
        else:
            return fn.partial(error.InvalidSymbolError, char = oper)

        arg = self.eval(arg)
        if hasattr(arg, 'with_traceback') or isinstance(arg, fn.partial):
            return arg
        
        ret = oper(arg)

        if not_none(ret):
            self.variables[self.var] = ret

    def function(self, string):
        matches = function.search(string).groups()
        indicies = list(filter(lambda a: not_none(matches[a]), range(len(matches))))
        mode = min(indicies)

        matches = list(filter(not_none, matches))

        if mode == 0:
            name, flags, code = matches
            flags = re.findall(r'[@#]{}[A-Za-z]+|[@{}]'.format('{1,2}', FLAGS), flags)

            switches = DEFAULT.copy()
            switches['@'] = list()

            for switch in flags:
                if switch[0] in '@#' and len(switch) > 1:
                    types, switch = re.findall(r'[@#]{1,2}|[A-Za-z]+', switch)
                    types = ''.join(sorted(types))
                    
                    if types == '#':
                        switches['@'].append(self.variables[switch])
                    if types == '@':
                        switches['@'].append({switch: Null()})
                    if types == '#@':
                        switches['@'].append({switch: self.variables[switch]})

                elif switch == '@':
                    switches['@'].append(Null())

                else:
                    if switch in switches.keys():
                        switches[switch] = True

            func = self.fn_shell(name, code, switches)
            self.functions[name] = func

        if mode == 3:
            self.lambdas += 1
            name = 'lambda {}'.format(self.lambdas)
            flags, code = matches

            switches = DEFAULT.copy()
            switches['@'] = list()
            switches['lambda'] = True
            
            for switch in flags:
                if switch in switches.keys():
                        switches[switch] = True
            
            func = self.fn_shell(name, code, switches)
            self.functions[name] = func

        if mode == 5 or mode == 7:
            func, args = matches
            args = list(map(lambda a: self.eval(a, True), splitting(args, '>')))

            for elem in args:
                if hasattr(elem, 'with_traceback'):
                    return elem
                if isinstance(elem, fn.partial) and hasattr(elem.func, 'with_traceback'):
                    return elem.func
            
            func = self.functions[func]
            ret = func(*args)
            
            if not_none(ret):
                self.variables[self.var] = ret
            self.called = True

    def eval(self, string, func = False):
        if not string:
            return None
        string = string.strip()
        
        if string == '?':
            try:
                return next(self.input)
            except StopIteration:
                return fn.partial(error.NoMoreInputError)

        if number.search(string):
            return eval(string)

        if string[0] == '"' and string[-1] == '"':
            valid = True
            
            for index, char in enumerate(string[1:-1]):
                if char == '"':
                    prev = string[index-1]
                    if prev != '\\':
                        valid = False

            if valid:
                return eval(string)

            return fn.partial(error.InvalidQuoteSyntaxError)

        if string[0] == "'" and string[-1] == "'":
            valid = True
            
            for index, char in enumerate(string[1:-1]):
                if char == "'":
                    prev = string[index-1]
                    if prev != '\\':
                        valid = False

            if valid:
                return eval(string)

            return fn.partial(error.InvalidQuoteSyntaxError)

        if string[0] == '[' and string[-1] == ']':
            return eval(string.replace(' ', ', '))

        if string in self.variables.keys():
            return self.variables[string]

        if string in self.functions.keys() and func:
            return self.functions[string]

        if string == '_':
            return self.preserve

        return fn.partial(error.InvalidArgumentError, arg = string)

    @property
    def commands(self):
        return {

            # Infix operators

            '+' : op.add,
            '-' : op.sub,
            '*' : op.mul,
            '/' : op.truediv,
            '\\': op.floordiv,
            '^' : pow,
            '>' : op.gt,
            '<' : op.lt,
            '%' : op.mod,
            '=' : op.eq,
            '|' : op.ne,
            'R' : random.randint,
            '@' : lambda x, y: print(x, end = str(y)),

            # Postfix operators

            '!' : math.factorial,
            '#' : op.neg,
            'N' : op.not_,
            'B' : bool,
            'O' : print,
            'o' : fn.partial(print, end = ''),
            'H' : nest(print, chr),
            'h' : nest(fn.partial(print, end = ''), chr),
            'd' : fn.partial(print, end = ' '),
            '_' : lambda _: self.preserve

        }

    @property
    def cons(self):
        return {

            'W': whileloop,
            'F': forloop,
            'D': doloop,
            'E': each,
            'I': ifstatement,

        }

def repl(history = None):
    if history is None:
        history = []

    codeprompt = '|>\t'
    argvprompt = '>>\t'
    separator  = '<; === ;>'

    code = [input(codeprompt)]
    while code[-1]:
        code.append(input(codeprompt))
    code.pop()
    code = '\n'.join(code)
    print()

    argv = [input(argvprompt)]
    while argv[-1]:
        argv.append(input(argvprompt))
    argv.pop()
    print()

    try:
        Script(code, argv, history = history)
        history.append((code, argv))
    except Exception as e:
        print(e, file = sys.stderr)

    print('\n' + separator + '\n')
    repl(history.copy())
    
addpp.Script = Script
addpp.VerboseScript = VerboseScript

if __name__ == '__main__':

    if len(sys.argv) == 1 or (sys.argv[1] in ('-r', '--repl')):
        try:
            repl()
        except KeyboardInterrupt:
            sys.exit(1)
            
        sys.exit(0)
    
    parser = argparse.ArgumentParser(prog = './add++')

    a = 'store_true'

    getcode = parser.add_mutually_exclusive_group()
    getcode.add_argument('-f', '--file', help = 'Specifies that code be read from a file', action = a)
    getcode.add_argument('-c', '--cmd', '--cmdline', help = 'Specifies that code be read from the command line', action = a)
    getcode.add_argument('-r', '--repl', help = 'Run code in a REPL environment', action = a)

    parser.add_argument('-e', '--error', help = 'Show full error messages', action = a)
    parser.add_argument('-i', '--implicit', help = 'Implicitly call a function at the end', action = a)
    parser.add_argument('-t', '--tokens', help = 'Show function tokens', action = a)
    parser.add_argument('-u', '--utf', help = 'Use utf-8 encoding rather than Add++', action = a)
    parser.add_argument('-vo', '--verbose-out', help = 'Output golfed code from verbose', action = a)
    parser.add_argument('-vh', '--version-help', help = 'Output all versions available', action = a)
    parser.add_argument('-o', '--suppress', help = 'Suppress output', action = a)
    parser.add_argument('-d', '--debug', help = 'Output information in vanilla mode\nWorks for versions 5.5 or greater', action = a)
    
    parser.add_argument('--version', help = 'Specify version to use', metavar = 'VERSION')
    parser.add_argument('--specify', help = 'Specify implicit function', metavar = 'FUNCTION')

    verbose = parser.add_mutually_exclusive_group()
    verbose.add_argument('-va', '--verbose-all', help = 'Make all sections verbose', action = a)
    verbose.add_argument('-vv', '--verbose-van', help = 'Make vanilla code verbose', action = a)
    verbose.add_argument('-vf', '--verbose-fun', help = 'Make function code verbose', action = a)

    parser.add_argument('program')
    parser.add_argument('input', nargs = '*', type = eval_)
    settings = parser.parse_args()

    if settings.repl:
        repl()

    if settings.version_help:
        print(*sorted(filter(
                lambda a: a not in ['__init__.py', 'error.py', 'extensions.py'] and a.endswith('.py'),
                os.listdir('./vers'))),
              sep = '\n', file = sys.stderr)

    if settings.version:
        settings.version = convert_version(settings.version)

    if settings.version:
        settings.verfile, settings.vernum = settings.version
        del settings.version
    else:
        settings.verfile, settings.vernum = None, VERSION
        del settings.version

    settings.verbose = [settings.verbose_all, settings.verbose_out,
                        settings.verbose_fun, settings.verbose_van]
    settings.useverbose = any(settings.verbose)

    if settings.vernum != VERSION:
        addpp = __import__(settings.verfile)
        addpp = eval('addpp.{}'.format(settings.verfile.split('.')[-1]))

    if settings.debug and settings.vernum < 5.5:
        raise NotImplementedError('The --debug flag only works on versions 5.5 or greater')

    if settings.cmd:
        code = settings.program
    elif settings.file:
        with open(settings.program, mode = 'rb') as file:
            contents = file.read()
        if 0 <= settings.vernum < 4.3 or settings.utf:
            code = contents.decode('utf-8')
        else:
            code = ''
            for ordinal in contents:
                code += addpp.code_page[ordinal]

    if settings.useverbose and settings.vernum >= 5:
        executor = lambda c, i, f, t, d: addpp.VerboseScript(c, i, f, t, d,
                                                *(settings.verbose + [settings]))
    else:
        executor = addpp.Script

    settings.code = code.replace('\r\n', '\n')

    if settings.suppress:
        with suppress_output():
            initiate(settings, executor)
    else:
        initiate(settings, executor)
