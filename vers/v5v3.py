import argparse
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

GLOBALREGISTER = None
VERSION = 5.3

identity = lambda a: a

class addpp(object):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)

addpp.code_page = '''€§«»Þþ¦¬£\t\nªº\r↑↓¢Ñ×¡¿ß‽⁇⁈⁉ΣΠΩΞΔΛ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~'''

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def process(lines):
    final = ['']
    for line in lines:
        if line.startswith(('  ', '\t')):
            final[-1] += line.split(';')[0].strip()
        else:
            final.append(line.split(';')[0].strip())
    return list(filter(None, map(lambda a: a.strip(','), final)))

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

def convert_version(version):
    # In  : 3
    # Out : vers/v3.py
    # In  : 4.3
    # Out : vers/v4/v4.3.py
    # In  : 2.5.1
    # Out : vers/v2/v2.5/v2.5.1.py

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
   if type(array) == list:
       for item in array:
           flat += flatten_array(item)
   else:
       flat.append(array)
   return flat

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
        executor(settings.code, settings.input,
                 [settings.implicit, settings.specify],
                 settings.tokens)
    else:
        try:
            executor(settings.code, settings.input,
                     [settings.implicit, settings.specify],
                     settings.tokens)
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

def VerboseScript(code, inputs, function, tokens, vall, vvan, vfun, vout, settings):
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
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'Null'

    def __str__(self):
        return 'Null'

class StackScript:

    def __call__(self, code):
        args = [self.name, code, self.args, self.functions,
                Stack(self.args), self.line, self.outer, self.tokens]
        return StackScript(*args)

    def __init__(self, name, code, args, funcs, stack,
                 line, outer, tokens, recur = False):
        
        self.args = args
        self.register = args if args else 0
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
        
        self.code = self.tokenize(code + ' ', tokens)
        
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
                    raise error.UnableToRetrieveFunctionError(line, outer[line-1], cmd[1:-1])
                
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
                    raise error.IncongruentTypesError(line, outer[line-1], cmd)
                
                except:
                    raise error.EmptyStackError(line, outer[line-1])
                    
                if result == Null:
                    raise error.InvalidSymbolError(line, outer[line-1], cmd)

                if type(result) == Stack:
                    self.stack.clear()
                    self.stack.push(*result)

                elif result is not None and result != []:
                    self.stack.push(result)

    def runquick(self, quick, cmd):
        if len(cmd) == 1:
            cmd = cmd[0]

        if cmd[0] == '{' and cmd[-1] == '}':
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
        stemp = ''
        ctemp = ''
        num = ''
        instr = False
        incall = False
        text = text.replace('{', ' {').replace('}', '} ')
        
        for i, char in enumerate(text):

            if char == '¡':
                if self.recur:
                    break
                else:
                    continue
            
            if char == '"': instr = not instr
            if char == '{': incall = True
            if char == '}': incall = False; ctemp += '}'; continue
	    
            if instr: stemp += char
            elif incall:ctemp += char
            else:
                if stemp:
                    final.append(stemp)
                    stemp = ''
                if ctemp:
                    final.append(ctemp)
                    ctemp = ''
                if isdigit(char):
                    try:
                        if char == '-':
                            if text[text.index(char)+1].isdigit():
                                num += char
                        else:
                            num += char
                    except: final.append(char)
                else:
                    if num:
                        final.append(num)
                        num = ''
                    final.append(char)

        if stemp: final.append(stemp)
        if ctemp: final.append(ctemp)
        if num: final.append(num)
        
        tokens = []
        for i, f in enumerate(final):
            if f in 'Bb':
                tokens.append(f + final.pop(i + 1))
            elif f in '" ':
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

                    if char in 'Bb':
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
                'F': ( 0, lambda: self.stack.push(*self.factors())      ),
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
                '~': (-1, lambda: Null,                                 ),

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
                'b<':(-1, lambda: Null                                   ),
                'b=':( 1, lambda x: self.eq(*x)                          ),
                'b>':(-1, lambda: Null                                   ),
                'b?':(-1, lambda: Null                                   ),
                'b@':(-1, lambda: Null                                   ),

                'bA':(-1, lambda: Null                                   ),
                'bB':( 0, lambda: self.pad_bin()                         ),
                'bC':(-1, lambda: Null                                   ),
                'bD':(-1, lambda: Null                                   ),
                'bE':(-1, lambda: Null                                   ),
                'bF':( 1, lambda x: self.flatten(x)                      ),
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
        
    def collect(self):
        array = []
        sub_array = []
        for element in self.stack:
            if type(element) == list:
                if sub_array:
                    array.append(sub_array)
                    sub_array = []
                array.append(element)
            else:
                sub_array.append(element)
        if sub_array:
            array.append(sub_array)
        self.stacks[self.index] = Stack(array)
		
    def columns(self):
        self.stacks[self.index] = Stack(map(list, zip(*self.stack)))

    def decrement(self):
        self.index -= 1
        
    def eq(self, *args):
        incs = [args[i] == args[i-1] for i in range(1, len(args))]
        return int(all(incs))
    
    def factors(self):
        lof = []
        x = self.stack.pop()
        if type(x) == str:
            return list(x)
        for i in range(1,int(x)):
            if x%i == 0:
                lof.append(i)
        return lof
	
    def flatten(self):
       copy = flatten_array(list(self.stack))
       self.stack.clear()
       self.stack.push(*copy)

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
	
    def run(self,flag,text):
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
		
    def wrap(self):
        array = self.stack.copy()
        self.stack.clear()
        self.stack.push(array)

class Function:

    def __init__(self, name, args, code, flags,
                 line = 0, g_code = "", outerf = {},
                 tkns = False):
        
        self.name = name
        self.args = args if args != -1 else 0
        self.lamb = args == -1
        self.code = code
        self.stack = Stack()
        self.flags = list(flags)
        self.line = line
        self.gen = g_code
        self.outerf = outerf
        self.tkns = tkns

    def __call__(self, *args, funccall = False, recur = False):
        if not self.flags[2]:
            args = list(args)[:self.args]
            while len(args) != self.args:
                args.append(-1)

        if self.flags[6]:
            args = args[::-1]
                
        if self.flags[4]:
            self.stack.push(list(args))
        else:
            self.stack.push(*args)

        if self.flags[5]:
            arr = []
            for element in self.stack:
                if hasattr(element, '__iter__'):
                    for i in element: arr.append(i)
                else:
                    arr.append(element)
            self.stack = Stack(arr.copy())
            
        script = StackScript(self.name, self.code, args, self.outerf,
                             self.stack, self.line, self.gen, self.tkns, recur)
        value = script.run(*self.flags[:2])
        self.stack = Stack()
        
        if self.flags[3]:
            print(value)
            if funccall:
                return value
            else:
                return Null(value)
        return int(value) if type(value) == bool else value
        
    def __repr__(self):
        return "<Function ${}: '{}'>".format(self.name, self.code)

class Script:

    def __init__(self,code, inputs, impfunc, tokens):

        self.NILADS = r'!~&#@NPOHSQVG'
        self.MONADS = r'+-*/\^><%=R'
        self.CONSTRUCTS = 'FWEIDL'
        self.FLAGS = r'*^?:!~#'

        self.code = process(code.split('\n'))

        self.called = False
        self.implicit = False
        self.stored = []
        self.string = ''
        self.functions = {}
        self.y = 0
        self.x = 0
        self.line = 0
        self.I = 0
        self.inputs = inputs
            
        for cmd in self.code:
            self.line += 1
            if type(self.x) == list:
                self.x = Stack(self.x)
            if type(self.y) == list:
                self.y = Stack(self.y)
                
            if cmd[0] in self.CONSTRUCTS:
                
                if cmd[:2] == 'EX':
                    loop = cmd.split(',')[1:]
                    for element in self.stored:
                        for chunk in loop:
                            self.run_chunk(chunk, x=element)
                            
                elif cmd[:2] == 'EY':
                    loop = cmd.split(',')[1:]
                    for element in self.stored:
                        for chunk in loop:
                            self.run_chunk(chunk, y=element)
                            
                elif cmd[:2] == 'W=':
                    loop = cmd.split(',')[1:]
                    while self.x == self.y:
                        for chunk in loop:
                            self.run_chunk(chunk)
                            
                elif cmd[:2] == 'W!':
                    loop = cmd.split(',')[1:]
                    while self.x != self.y:
                        for chunk in loop:
                            self.run_chunk(chunk)
                            
                elif cmd[0] == 'F':
                    loop = cmd.split(',')[1:]
                    for _ in range(self.x):
                        for chunk in loop:
                            self.run_chunk(chunk)
                            
                elif cmd[0] == 'I':
                    loop = cmd.split(',')[1:]
                    if self.x:
                        for chunk in loop:
                            self.run_chunk(chunk)
                            
                elif cmd[0] == 'W':
                    loop = cmd.split(',')[1:]
                    while self.x:
                        for chunk in loop:
                            self.run_chunk(chunk)
                            
                elif cmd[0] == 'D':
                    cmd = cmd.split(',')
                    func_name = cmd[1]
                    func_args = cmd[2].count('@')
                    func_flags = []
                    for flag in self.FLAGS:
                        func_flags.append(flag in cmd[2])
                    func_code = ','.join(cmd[3:])
                    self.functions[func_name] = Function(func_name, func_args, func_code, func_flags, self.line, code, self.functions, tokens)

                elif cmd[0] == 'L':
                    cmd = cmd.split(',')
                    flags = cmd[0][1:]
                    lambda_c = ','.join(cmd[1:])
                    lambda_n = len(list(filter(lambda a: bool(re.search(r'^lambda \d+$', a)), self.functions.keys()))) + 1
                    name = 'lambda {}'.format(lambda_n)
                    lambda_f = []
                    for flag in self.FLAGS:
                        lambda_f.append(flag == '?' or flag in flags)
                    self.functions[name] = Function(name, -1, lambda_c, lambda_f, self.line, code, self.functions, tokens)
                    
            else:
                self.implicit = True
                if cmd[:2] in ['x:', 'y:']:
                    if cmd[0] == 'x': acc = self.x; acc_n = 'x'
                    else: acc = self.y; acc_n = 'y'
                        
                    c = cmd[2:]
                    if c == '?':
                        try: acc = self.inputs[self.I]; self.I += 1
                        except: acc = 0
                    elif c == 'G':
                        try: acc = self.stored.pop()
                        except: raise error.EmptySecondStackError(self.line, self.code[self.line-1])
                    elif c == 'x': acc = self.x
                    elif c == 'y': acc = self.y
                    elif c == 'g': acc = self.stored[-1]
                    else: acc = eval_(c)
					
                    if acc_n == 'x': self.x = acc
                    if acc_n == 'y': self.y = acc
                        
                elif cmd[0] == '$':
                    self.called = True
                    cmd = cmd.split('>')
                    try: func = self.functions[cmd[0][1:]]
                    except: raise error.UnableToRetrieveFunctionError(self.line, self.code[self.line-1], cmd[0][1:])
                    args = []
                    for c in cmd[1:]:
                        if c == '?':
                            try: args.append(self.inputs[self.I]); self.I += 1
                            except: args.append(0)
                        elif c == 'G':
                            try: args.append(self.stored.pop())
                            except: raise error.EmptySecondStackError(self.line, self.code[self.line-1])
                        elif c == 'x': args.append(self.x)
                        elif c == 'y': args.append(self.y)
                        elif c == 'g': args.append(self.stored[-1])
                        elif c == '_': args += self.stored
                        else: args.append(eval_(c))
                            
                    value = func(*args)
                    if type(value) == Null: value = value.value
                    if type(value) == str: self.stored.append(value)
                    if type(value) == list:
                        for v in value:
                            self.stored.append(v)
                    self.x = value
                    
                else:
                    self.run_chunk(cmd)
                    
        if impfunc[0] and not self.called and self.functions:
            
            if impfunc[1]:
                func = self.functions[impfunc[1]]
            else:
                func = self.functions[list(self.functions.keys())[0]]
                
            if self.I < len(self.inputs):
                result = func(*self.inputs[self.I:])
            elif self.x:
                if self.y:
                    result = func(self.x, self.y)
                else:
                    result = func(self.x)
            else:
                result = func()
                
            if type(result) != Null and not self.implicit:
                print(result)
                
    def run_chunk(self, cmd, x=None, y=None):
        if x is not None: self.x = x
        if y is not None: self.y = y

        symbol = cmd[0]
        if symbol == "_":
            for i in self.inputs:
                self.stored.append(i)
        if symbol == '}': self.x, self.y = self.y, self.x
            
        if len(cmd) > 1: value = eval_(cmd[1:])
        else: value = None

        if cmd[:2] in ['x:', 'y:']:
            if cmd[0] == 'x': acc = self.x; acc_n = 'x'
            else: acc = self.y; acc_n = 'y'
                
            c = cmd[2:]
            if c == '?':
                try: acc = self.inputs[self.I]; self.I += 1
                except: acc = 0
            elif c == 'G':
                try: acc = self.stored.pop()
                except: raise error.EmptySecondStackError(self.line, self.code[self.line-1])
            elif c == 'x': acc = self.x
            elif c == 'y': acc = self.y
            elif c == 'g': acc = self.stored[-1]
            else: acc = eval_(c)
                                
            if acc_n == 'x': self.x = acc
            if acc_n == 'y': self.y = acc
        elif symbol == '$':
            self.called = True
            cmd = cmd.split('>')
            try: func = self.functions[cmd[0][1:]]
            except: raise error.UnableToRetrieveFunctionError(self.line, self.code[self.line-1], cmd[0][1:])
            args = []
            for c in cmd[1:]:
                if c == '?':
                    try: args.append(self.inputs[self.I]); self.I += 1
                    except: args.append(0)
                elif c == 'G':
                    try: args.append(self.stored.pop())
                    except: raise error.EmptySecondStackError(self.line, self.code[self.line-1])
                elif c == 'x': args.append(self.x)
                elif c == 'y': args.append(self.y)
                elif c == 'g': args.append(self.stored[-1])
                elif c == '_': args += self.stored
                else: args.append(eval_(c))
                    
            value = func(*args)
            if type(value) == Null: value = value.value
            if type(value) == str: self.stored.append(value)
            if type(value) == list:
                for v in value:
                    self.stored.append(v)
            self.x = value
        elif value is not None:
            if value == '?':
                try: value = self.inputs[self.I];  self.I += 1
                except: raise error.NoMoreInputError(self.line, self.code[self.line-1])
            if value == 'G':
                try: value = self.stored.pop()
                except: raise error.EmptySecondStackError(line, self.code[line-1])
            if value == 'g': value = self.stored[-1]
            if value == 'x': value = self.x
            if value == 'y': value = self.y
                
            try: self.x = self.COMMANDS[symbol](value)
            except ZeroDivisionError: raise error.DivisionByZeroError(self.line, self.code[self.line-1])
            except: raise error.InvalidSymbolError(self.line, self.code[self.line-1], symbol)
                
        else:
            try: v = self.COMMANDS[symbol]()
            except:
                if symbol == '?':
                    try: v = self.inputs[self.I]; self.I += 1
                    except: raise error.NoMoreInputError(self.line, self.code[self.line-1])
                else: v = None
            if v is None: return
            self.x = v

    def __call__(self, *values):
        return None

    @property
    def COMMANDS(self):
        return {'+':self.add,
                '-':self.minus,
                '*':self.times,
                '/':self.divide,
                '\\':self.int_divide,
                '^':self.power,
                '>':self.greater,
                '<':self.less,
                '!':self.factorial,
                '%':self.modulo,
                '~':self.negative,
                '=':self.equal,
                '&':self.next,
                '#':self.double,
                '@':self.half,
                '|':self.notequals,
                
                'N':self.not_,
                'P':self.print,
                'O':self.print_,
                'H':self._print,
                'R':self.randint,
                'S':self.sqrt,
                'V':self.store,
                'G':self.get,
                'U':self.globalget,
                'K':self.globalsave,

                'F':self,
                'I':self,
                'W':self,
                'E':self,
                'D':self,
                'L':self}

    def add(self, y):        return self.x + y
    def minus(self, y):      return self.x - y
    def times(self, y):      return self.x * y
    def divide(self, y):     return self.x / y
    def int_divide(self, y): return int(self.x / y)
    def power(self, y):      return self.x ** y
    def greater(self, y):    return int(self.x > y)
    def less(self, y):       return int(self.x < y)
    def factorial(self):     return math.factorial(self.x)
    def modulo(self, y):     return self.x % y
    def negative(self):      return -self.x
    def equal(self, y):      return int(self.x == y)
    def next(self):          self.string += chr(self.x)
    def double(self):        return self.x * 2
    def half(self):          return self.x / 2
    def notequals(self, y):  return int(self.x != y)
    def not_(self):          return int(not self.x)
    def print_(self):        print(self.x)
    def sqrt(self):          return math.sqrt(self.x)
    def store(self):         self.stored.append(self.x)
    def get(self):           self.x = self.stored.pop()

    def globalsave(self):
        global GLOBALREGISTER
        GLOBALREGISTER = self.x
        
    def globalget(self):     self.x = GLOBALREGISTER
        
    def _print(self):
        if self.string: print(end=self.string)
        else: print(end=chr(self.x))
            
    def print(self):
        if self.string: print(self.string)
        else: print(chr(self.x))
            
    def randint(self,y=0):
        if y > self.x: return random.randint(self.x, y)
        return random.randint(y, self.x)

addpp.Script = Script
addpp.VerboseScript = VerboseScript

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog = './add++')

    a = 'store_true'

    getcode = parser.add_mutually_exclusive_group()
    getcode.add_argument('-f', '--file', help = 'Specifies that code be read from a file', action = a)
    getcode.add_argument('-c', '--cmd', '--cmdline', help = 'Specifies that code be read from the command line', action = a)

    parser.add_argument('-e', '--error', help = 'Show full error messages', action = a)
    parser.add_argument('-i', '--implicit', help = 'Implicitly call a function at the end', action = a)
    parser.add_argument('-t', '--tokens', help = 'Show function tokens', action = a)
    parser.add_argument('-u', '--utf', help = 'Use utf-8 encoding rather than Add++', action = a)
    parser.add_argument('-vo', '--verbose-out', help = 'Output golfed code from verbose', action = a)
    parser.add_argument('-vh', '--version-help', help = 'Output all versions available', action = a)
    parser.add_argument('-o', '--suppress', help = 'Suppress output', action = a)
    
    parser.add_argument('--version', help = 'Specify version to use', metavar = 'VERSION')
    parser.add_argument('--specify', help = 'Specify implcit function', metavar = 'FUNCTION')

    verbose = parser.add_mutually_exclusive_group()
    verbose.add_argument('-va', '--verbose-all', help = 'Make all sections verbose', action = a)
    verbose.add_argument('-vv', '--verbose-van', help = 'Make vanilla code verbose', action = a)
    verbose.add_argument('-vf', '--verbose-fun', help = 'Make function code verbose', action = a)

    parser.add_argument('program')
    parser.add_argument('input', nargs = '*', type = eval_)
    settings = parser.parse_args()

    if settings.version_help:
        print(*list(filter(
                lambda a: a not in ['__init__.py', 'error.py'] and a.endswith('.py'),
                os.listdir('vers'))),
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
        executor = lambda c, i, f, t: addpp.VerboseScript(c, i, f, t,
                                                *(settings.verbose + [settings]))
    else:
        executor = addpp.Script

    settings.code = code.replace('\r\n', '\n')

    if settings.suppress:
        with suppress_output():
            initiate(settings, executor)
    else:
        initiate(settings, executor)
