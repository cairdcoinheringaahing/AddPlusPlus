import argparse
import functools as fn
import itertools as it
import math
import operator as op
import random
import re
import sys

import error

GLOBALREGISTER = None
VERSION = 4.5

class addpp(object):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)

addpp.code_page = '''€§«»Þþ¦¬£\t\nªº\r↑↓¢Ñ×¡¿ß‽⁇⁈⁉ΣΠΩΞΔΛ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~'''

def isdigit(string):
    return all(i in '1234567890-.' for i in string)

def eval_(string):
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

    def __init__(self, code, args, funcs, stack, line, outer, tokens):
        self.args = args
        self.register = args if args else 0
        self.stacks = [stack]
        self.index = 0
        self.quicks = ''.join(list(self.QUICKS.keys()))
        self.code = self.tokenize(code + ' ', tokens)
        self.prevcall = None
        self.functions = funcs
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
                quick, cmd = cmd
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
                    
                self.prevcall = func(*feed, funccall = True)
                self.stack.push(self.prevcall)
                
            elif isdigit(cmd):
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
                    result = command(*[self.stack.pop() for _ in range(arity)])
                except TypeError:
                    raise error.IncongruentTypesError(line, outer[line-1], cmd)
                except:
                    raise error.EmptyStackError(line, outer[line-1])
                    
                if result == Null:
                    raise error.InvalidSymbolError(line, outer[line-1], cmd)

                if result is not None and result != []:
                        self.stack.push(result)

    def runquick(self, quick, cmd):
            if cmd in self.QUICKS.keys():
                self.runquick(*cmd)
            elif cmd[0] == '{' and cmd[-1] == '}':
                func = self.functions[cmd[1:-1]]
                self.QUICKS[quick]((func.args, func))
            else:
                self.QUICKS[quick](self.COMMANDS[cmd.strip()])

    def tokenize(self, text, output):
        
        final = []
        stemp = ''
        ctemp = ''
        num = ''
        instr = False
        incall = False
        text = text.replace('{', ' {').replace('}', '} ')
        
        for i, char in enumerate(text):
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

        chain = [[]]

        index = 0
        while index < len(tokens):
            if tokens[index] in self.quicks:
                if type(chain[-1]) != list:
                    chain.append([])
                chain[-1] += [tokens[index], tokens[index+1]]
                index += 1
            else:
                chain.append(tokens[index])
            index += 1

        if output:
            print(chain)
        return chain

    @property
    def QUICKS(self):
        return {
                '€': self.quickeach,
                '§': self.quicksort,
                '«': self.quickmax,
                '»': self.quickmin,
                'Þ': self.quickfiltertrue,
                'þ': self.quickfilterfalse,
                '¦': self.quickreduce,
                '¬': self.quickaccumulate,
                '£': self.quickstareach,
                'ª': self.quickall,
                'º': self.quickany,
                '↑': self.quicktakewhile,
                '↓': self.quickdropwhile,
                '¢': self.quickgroupby,
                'Ñ': self.quickneighbours,
               }
    
    @property
    def COMMANDS(self):
        return {
                '!': (1, lambda x: not x                                ),
                '#': (0, lambda: self.stack.sort()                      ),
                '$': (0, lambda: self.stack.swap()                      ),
                '%': (2, lambda x, y: modulo(x, y)                      ),
                '&': (2, lambda x, y: x and y                           ),
                "'": (1, lambda x: x * 2                                ),
                '(': (0, lambda: self.decrement()                       ),
                ')': (0, lambda: self.increment()                       ),
                '*': (2, lambda x, y: multiply(x, y)                    ),
                '+': (2, lambda x, y: add(x, y)                         ),
                '/': (2, lambda x, y: divide(x, y)                      ),
                ':': (0, lambda: Null                                   ),
                '<': (2, lambda x, y: x < y                             ),
                '=': (2, lambda x, y: x == y                            ),
                '>': (2, lambda x, y: x > y                             ),
                '?': (1, lambda x: (x > 0) - (x < 0)                    ),
                '@': (0, lambda: self.stack.reverse()                   ),
                
                'A': (0, lambda: self.stack.push(*self.args)            ),
                'B': (1, lambda x: self.stack[:x]                       ),
                'C': (1, lambda x: chr(x)                               ),
                'D': (1, lambda x: self.stack[-x]                       ),
                'E': (1, lambda x: list(map(list, enumerate(x)))        ),
                'F': (0, lambda: self.stack.push(*self.factors())       ),
                'G': (0, lambda: self.stack.push(self.register)         ),
                'H': (0, lambda: print(''.join(map(str, self.stack)))   ),
                'I': (0, lambda: Null                                   ),
                'J': (0, lambda: self.join('')                          ),
                'K': (0, lambda: Null                                   ),
                'L': (0, lambda: len(self.stack)                        ),
                'M': (0, lambda: max(self.stack)                        ),
                'N': (0, lambda: '\n'.join(map(str, self.stack))        ),
                'O': (1, lambda x: ord(x)                               ),
                'P': (1, lambda x: isprime(x)                           ),
                'R': (1, lambda x: list(range(1, x+1))                  ),
                'S': (0, lambda: self.remove_duplicates()               ),
                'T': (0, lambda: Null                                   ),
                'U': (0, lambda: Null                                   ),
                'V': (1, lambda x: self.store(x)                        ),
                'X': (2, lambda x, y: [x for _ in range(y)]             ),
                'Y': (0, lambda: Null                                   ),
                'Z': (0, lambda: Null                                   ),

                '[': (0, lambda: self.prevcall                          ),
                ']': (1, lambda x: self.run_lambda(x)                   ),
                '^': (2, lambda x, y: exponent(x, y)                    ),
                '_': (2, lambda x, y: subtract(x, y)                    ),
                '`': (0, lambda: Null                                   ),

                'a': (0, lambda: list(self.args)                        ),
                'b': (0, lambda: Null                                   ),
                'c': (0, lambda: self.stack.clear()                     ),
                'd': (0, lambda: self.stack[-1]                         ),
                'e': (2, lambda x, y: x in y                            ),
                'f': (0, lambda: list(filter(isprime, self.factors()))  ),
                'g': (0, lambda: Null                                   ),
                'h': (0, lambda: print(self.stack)                      ),
                'i': (1, lambda x: int(x)                               ),
                'j': (1, lambda x: self.join(str(x))                    ),
                'k': (0, lambda: Null                                   ),
                'l': (0, lambda: Null                                   ),
                'm': (0, lambda: min(self.stack)                        ),
                'n': (0, lambda: self.join()                            ),
                'o': (2, lambda x, y: x or y                            ),
                'p': (1, lambda x: None                                 ),
                'q': (1, lambda x: set(x)                               ),
                'r': (2, lambda x, y: list(range(x, y))                 ),
                's': (0, lambda: sum(self.stack)                        ),
                't': (2, lambda x, y: str(x).split(str(y))              ),
                'u': (0, lambda: Null                                   ),
                'v': (1, lambda x: eval(x)                              ),
                'w': (0, lambda: Null                                   ),
		'x': (1, lambda x: [self.stack[-1] for _ in range(x)]   ),
		'y': (1, lambda x: [self.stack.push(self.stack[-1]) for _ in range(x)][:0]          ),
                'z': (0, lambda: Null                                   ),
                
                '|': (1, lambda x: abs(x)                               ),
                '~': (0, lambda: Null                                   ),

                'B!':(0, lambda: self.a(op.not_)                        ),
                'B#':(0, lambda: self.a(sorted)                         ),
                'B$':(0, lambda: Null                                   ),
                'B%':(0, lambda: self.a(lambda l: fn.reduce(op.mod, l)) ),
                'B&':(0, lambda: self.a(lambda l: fn.reduce(op.and_, l))),
                "B'":(0, lambda: self.a(lambda x: 2 * x)                ),
                'B(':(0, lambda: self.stacks[self.index - 1].pop()      ),
                'B)':(0, lambda: self.stacks[(self.index + 1) % len(self.stacks)].pop()             ),
                'B*':(0, lambda: self.a(lambda l: fn.reduce(op.mul, l)) ),
                'B+':(0, lambda: self.a(lambda l: fn.reduce(op.add, l)) ),
                'B/':(0, lambda: self.a(lambda l: fn.reduce(op.truediv, l))                         ),
                'B:':(0, lambda: Null                                   ),
                'B<':(0, lambda: Null                                   ),
                'B=':(0, lambda: self.a(lambda l: self.eq(*l))          ),
                'B>':(0, lambda: Null                                   ),
                'B?':(0, lambda: Null                                   ),
                'B@':(0, lambda: self.a(reversed, True)                 ),
                
                'BA':(0, lambda: self.a(abs)                            ),
                'BB':(1, lambda x: base(x, 2)                           ),
                'BC':(0, lambda: self.collect()                         ),
                'BD':(0, lambda: self.a(lambda i: list(map(int, str(i))))                           ),
                'BE':(0, lambda: self.a(lambda i: i in self.stack[-1])  ),
                'BF':(0, lambda: self.flatten()                         ),
                'BG':(0, lambda: Null                                   ),
                'BH':(0, lambda: Null                                   ),
                'BI':(0, lambda: Null                                   ),
                'BJ':(0, lambda: self.a(lambda i: ''.join(map(str, i))) ),
                'BK':(0, lambda: GLOBALREGISTER                         ),
                'BL':(0, lambda: self.a(len)                            ),
                'BM':(0, lambda: self.a(max)                            ),
                'BN':(0, lambda: Null                                   ),
                'BO':(0, lambda: Null                                   ),
                'BP':(0, lambda: self.a(lambda x: x[1:])                ),
                'BQ':(0, lambda: self.a(self.remove_duplicates)         ),
                'BR':(0, lambda: self.a(lambda x: list(range(1, x + 1)))),
                'BS':(0, lambda: Stack([self.stack[i : i+2] for i in range(len(self.stack) - 1)])   ),
                'BT':(0, lambda: Null                                   ),
                'BU':(0, lambda: Null                                   ),
                'BV':(1, lambda x: exec(x)                              ),
                'BW':(0, lambda: Stack([i for i in self.stack[:-1] if i not in self.stack[-1]])     ),
                'BX':(1, lambda x: random.choice(x)                     ),
                'BY':(0, lambda: self.a(random.choice)                  ),
                'BZ':(0, lambda: Stack(filter(None, self.stack))        ),

                'B]':(0, lambda: self.wrap()                            ),
                'B[':(0, lambda: self.a(lambda l: [l])                  ),
                'B^':(0, lambda: self.a(lambda l: fn.reduce(op.xor, l)) ),
                'B_':(0, lambda: self.a(lambda l: fn.reduce(op.sub, l)) ),
                'B`':(0, lambda: self.a(lambda l: fn.reduce(op.pow, l)) ),

                'Ba':(2, lambda x, y: x & y                             ),
                'Bb':(2, lambda x, y: unbase(x, y)                      ),
                'Bc':(0, lambda: self.columns()                         ),
                'Bd':(0, lambda: self.a(lambda l: fn.reduce(op.floordiv, l))                        ),
                'Be':(1, lambda x: [i in self.stack[-1] for i in x]     ),
                'Bf':(1, lambda x: ~x                                   ),
                'Bg':(0, lambda: Null                                   ),
                'Bh':(0, lambda: Null                                   ),
                'Bi':(0, lambda: self.a(int)                            ),
                'Bj':(0, lambda: self.a(isprime)                        ),
                'Bk':(1, lambda x: self.assign(x)                       ),
                'Bl':(0, lambda: Null                                   ),
                'Bm':(0, lambda: self.a(min)                            ),
                'Bn':(0, lambda: self.a(lambda i: -i)                   ),
                'Bo':(2, lambda x, y: x | y                             ),
                'Bp':(0, lambda: self.a(lambda x: x[:-1])               ),
                'Bq':(0, lambda: Null                                   ),
                'Br':(0, lambda: Null                                   ),
                'Bs':(0, lambda: self.a(sum)                            ),
                'Bt':(0, lambda: Null                                   ),
                'Bu':(0, lambda: Null                                   ),
                'Bv':(0, lambda: self.a(lambda i: int(''.join(map(str, i))))                        ),
                'Bw':(0, lambda: Stack([i for i in self.stack[:-1] if i in self.stack[-1]])         ),
                'Bx':(2, lambda x, y: x ^ y                             ),
                'By':(0, lambda: Null                                   ),
                'Bz':(0, lambda: Null                                   ),
                
                'B|':(0, lambda: self.a(lambda l: fn.reduce(op.or_, l)) ),
                'B~':(0, lambda: self.a(op.inv)                         ),
                
                'b!':(1, lambda x: list(map(lambda a: int(not a), x))   ),
                'b#':(0, lambda: Null                                   ),
                'b$':(0, lambda: Null                                   ),
                'b%':(1, lambda x: fn.reduce(op.mod, x)                 ),
                'b&':(1, lambda x: fn.reduce(op.and_, x)                ),
                "b'":(1, lambda x: [i * 2 for i in x]                   ),
                'b(':(1, lambda x: self.stacks[self.index - 1].push(x)  ),
                'b)':(1, lambda x: self.stacks[(self.index + 1) % len(self.stacks)].push(x)         ),
                'b*':(1, lambda x: fn.reduce(op.mul, x)                 ),
                'b+':(1, lambda x: fn.reduce(op.add, x)                 ),
                'b/':(1, lambda x: fn.reduce(op.truediv, x)             ),
                'b:':(0, lambda: Null                                   ),
                'b<':(0, lambda: Null                                   ),
                'b=':(1, lambda x: self.eq(*x)                          ),
                'b>':(0, lambda: Null                                   ),
                'b?':(0, lambda: Null                                   ),
                'b@':(0, lambda: Null                                   ),

                'bA':(0, lambda: Null                                   ),
                'bB':(0, lambda: self.pad_bin()                         ),
                'bC':(0, lambda: Null                                   ),
                'bD':(0, lambda: Null                                   ),
                'bE':(0, lambda: Null                                   ),
                'bF':(1, lambda x: self.flatten(x)                      ),
                'bG':(0, lambda: Null                                   ),
                'bH':(0, lambda: Null                                   ),
                'bI':(0, lambda: Null                                   ),
                'bJ':(0, lambda: Null                                   ),
                'bK':(0, lambda: Null                                   ),
                'bL':(1, lambda x: len(x)                               ),
                'bM':(1, lambda x: max(x)                               ),
                'bN':(0, lambda: Null                                   ),
                'bO':(0, lambda: Null                                   ),
                'bP':(0, lambda: Null                                   ),
                'bQ':(0, lambda: Null                                   ),
                'bR':(1, lambda x: x[::-1]                              ),
                'bS':(0, lambda: Null                                   ),
                'bT':(0, lambda: Null                                   ),
                'bU':(1, lambda x: self.stack.push(*x)                  ),
                'bV':(1, lambda x: self.selfexec(x)                     ),
                'bW':(0, lambda: Null                                   ),
                'bX':(0, lambda: Null                                   ),
                'bY':(0, lambda: Null                                   ),
                'bZ':(0, lambda: Null                                   ),

                'b[':(0, lambda: Null                                   ),
                'b]':(1, lambda x: [x]                                  ),
                'b^':(1, lambda x: fn.reduce(op.xor, x)                 ),
                'b_':(1, lambda x: fn.reduce(op.sub, x)                 ),
                'b`':(1, lambda x: fn.reduce(op.pow, x)                 ),

                'ba':(0, lambda: Null                                   ),
                'bb':(2, lambda x, y: base(x, y)                        ),
                'bc':(0, lambda: Null                                   ),
                'bd':(1, lambda x: fn.reduce(op.floordiv, x)            ),
                'be':(0, lambda: Null                                   ),
                'bf':(0, lambda: Null                                   ),
                'bg':(0, lambda: Null                                   ),
                'bh':(0, lambda: Null                                   ),
                'bi':(1, lambda x: x                                    ),
                'bj':(0, lambda: Null                                   ),
                'bk':(0, lambda: Null                                   ),
                'bl':(0, lambda: Null                                   ),
                'bm':(1, lambda x: min(x)                               ),
                'bn':(0, lambda: Null                                   ),
                'bo':(0, lambda: Null                                   ),
                'bp':(0, lambda: Null                                   ),
                'bq':(0, lambda: Null                                   ),
                'br':(0, lambda: Null                                   ),
                'bs':(0, lambda: Null                                   ),
                'bt':(0, lambda: Null                                   ),
                'bu':(0, lambda: Null                                   ),
                'bv':(0, lambda: Null                                   ),
                'bw':(0, lambda: Null                                   ),
                'bx':(0, lambda: Null                                   ),
                'by':(0, lambda: Null                                   ),
                'bz':(0, lambda: Null                                   ),
                
                'b|':(1, lambda x: fn.reduce(op.or_, x)                 ),
                'b~':(1, lambda x: list(map(op.inv, x))                 ),
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
        return all(incs)
    
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

    def quickaccumulate(self, cmd):
        arity, cmd = cmd
        self.stacks[self.index] = Stack(it.accumulate(self.stack, cmd))

    def quickall(self, cmd):
        arity, cmd = cmd
        if arity == 2:
            right = self.stack.pop()
            self.stack.push(all(map(cmd, it.repeat(right), self.stack)))
        if arity == 1:
            self.stack.push(all(map(cmd, self.stack)))

    def quickany(self, cmd):
        arity, cmd = cmd
        if arity == 2:
            right = self.stack.pop()
            self.stack.push(any(map(cmd, it.repeat(right), self.stack)))
        if arity == 1:
            self.stack.push(any(map(cmd, self.stack)))

    def quickdropwhile(self, cmd):
        arity, cmd = cmd
        if arity == 2:
            right = self.stack.pop()
            self.stacks[self.index] = Stack(it.dropwhile(lambda v: cmd(right, v), self.stack))
        if arity == 1:
            self.stacks[self.index] = Stack(it.dropwhile(cmd, self.stack))

    def quickeach(self, cmd):
        arity, cmd = cmd
        if arity == 2:
            right = self.stack.pop()
            self.stacks[self.index] = Stack(map(cmd, it.repeat(right), self.stack))
        if arity == 1:
            self.stacks[self.index] = Stack(map(cmd, self.stack))

    def quickfilterfalse(self, cmd):
        arity, cmd = cmd
        if arity == 2:
            right = self.stack.pop()
            self.stacks[self.index] = Stack(it.filterfalse(lambda v: cmd(right, v), self.stack))
        if arity == 1:
            self.stacks[self.index] = Stack(it.filterfalse(cmd, self.stack))

    def quickfiltertrue(self, cmd):
        arity, cmd = cmd
        if arity == 2:
            right = self.stack.pop()
            self.stacks[self.index] = Stack(filter(lambda v: cmd(right, v), self.stack))
        if arity == 1:
            self.stacks[self.index] = Stack(filter(cmd, self.stack))

    def quickgroupby(self, cmd):
        arity, cmd = cmd
        if arity == 2:
            right = self.stack.pop()
            self.stacks[self.index] = Stack(groupby(lambda a: cmd(right, a), self.stack))
        if arity == 1:
            self.stacks[self.index] = Stack(groupby(cmd, self.stack))

    def quickmax(self, cmd):
        arity, cmd = cmd
        if arity == 2:
            right = self.stack.pop()
            self.stack.push(max(self.stack, key = lambda v: cmd(v, right)))
        if arity == 1:
            self.stack.push(max(self.stack, key = cmd))

    def quickmin(self, cmd):
        arity, cmd = cmd
        if arity == 2:
            right = self.stack.pop()
            self.stack.push(min(self.stack, key = lambda v: cmd(v, right)))
        if arity == 1:
            self.stack.push(min(self.stack, key = cmd))

    def quickneighbours(self, cmd):
        arity, cmd = cmd
        sublists = [self.stack[i:i+2] for i in range(len(self.stack) - 1)]
        if arity == 2:
            self.stacks[self.index] = Stack([cmd(*sub) for sub in sublists])
        if arity == 1:
            self.stacks[self.index] = Stack([cmd(sub) for sub in sublists])

    def quickreduce(self, cmd):
        arity, cmd = cmd
        self.stack.push(fn.reduce(cmd, self.stack))

    def quicksort(self, cmd):
        arity, cmd = cmd
        if arity == 2:
            right = self.stack.pop()
            self.stacks[self.index] = Stack(sorted(self.stack, key = lambda v: cmd(right, v)))
        if arity == 1:
            self.stacks[self.index] = Stack(sorted(self.stack, key = cmd))

    def quickstareach(self, cmd):
        arity, cmd = cmd
        self.stacks[self.index] = Stack(it.starmap(cmd, self.stack))

    def quicktakewhile(self, cmd):
        arity, cmd = cmd
        if arity == 2:
            right = self.stack.pop()
            self.stacks[self.index] = Stack(it.takewhile(lambda v: cmd(right, v), self.stack))
        if arity == 1:
            self.stacks[self.index] = Stack(it.takewhile(cmd, self.stack))
		
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

    def __init__(self, name, args, code, line, g_code, outerf, tkns, *flags):
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

    def __call__(self, *args, funccall = False):
        if not self.flags[2]:
            args = list(args)[:self.args]
            while len(args) != self.args:
                args.append(-1)
                
        if self.flags[4]:
            self.stack.push(list(args))
        else:
            self.stack.push(*args)

        if self.flags[5]:
            arr = []
            for element in self.stack:
                if hasattr(arr, '__iter__'):
                    for i in element: arr.append(i)
                else:
                    arr.append(element)
            self.stack = Stack(arr.copy())
            
        script = StackScript(self.code, args, self.outerf, self.stack, self.line, self.gen, self.tkns)
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
        return '<Function ${}: {}>'.format(self.name, self.code)

class Script:

    def process(self, lines):
        final = ['']
        for line in lines:
            if line.startswith(('  ', '\t')):
                final[-1] += line.split(';')[0].strip()
            else:
                final.append(line.split(';')[0].strip())
        return list(filter(None, map(lambda a: a.strip(','), final)))

    def __init__(self,code, inputs, impfunc, tokens):

        self.NILADS = r'!~&#@NPOHSQVG'
        self.MONADS = r'+-*/\^><%=R'
        self.CONSTRUCTS = 'FWEIDL'
        self.FLAGS = r'*^?:!~'

        self.code = self.process(code.split('\n'))

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
                    self.functions[func_name] = Function(func_name, func_args, func_code, self.line, code, self.functions, tokens, *func_flags)

                elif cmd[0] == 'L':
                    cmd = cmd.split(',')
                    flags = cmd[0][1:]
                    lambda_c = ','.join(cmd[1:])
                    lambda_n = len(list(filter(lambda a: bool(re.search(r'^lambda \d+$', a)), self.functions.keys()))) + 1
                    name = 'lambda {}'.format(lambda_n)
                    lambda_f = []
                    for flag in self.FLAGS:
                        lambda_f.append(flag == '?' or flag in flags)
                    self.functions[name] = Function(name, -1, lambda_c, self.line, code, self.functions, tokens, *lambda_f)
                    
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
                    
        if impfunc and not self.called and self.functions:
            func = self.functions[list(self.functions.keys())[0]]
            if self.I < len(self.inputs): result = func(*self.inputs[self.I:])
            elif self.x:
                if self.y: result = func(self.x, self.y)
                else: result = func(self.x)
            else: result = func()
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
    parser.add_argument('--version', help = 'Specify version to use', metavar = 'VERSION')

    verbose = parser.add_mutually_exclusive_group()
    verbose.add_argument('-va', '--verbose-all', help = 'Make all sections verbose', action = a)
    verbose.add_argument('-vo', '--verbose-out', help = 'Output golfed code from verbose', action = a)
    verbose.add_argument('-vv', '--verbose-van', help = 'Make vanilla code verbose', action = a)
                         
    parser.add_argument('program')
    parser.add_argument('input', nargs = '*', type = eval_)
    settings = parser.parse_args()

    if settings.version:
        settings.version = convert_version(settings.version)
        print(settings.version)

    if settings.version:
        settings.verfile, settings.vernum = settings.version
        del settings.version
    else:
        settings.verfile, settings.vernum = None, VERSION
        del settings.version

    settings.verbose = any([settings.verbose_all, settings.verbose_out, settings.verbose_van])

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

    if settings.verbose and settings.vernum >= 5:
        executor = addpp.VerboseScript
    else:
        executor = addpp.Script

    code = code.replace('\r\n', '\n')

    if settings.error:
        executor(code, settings.input, settings.implicit, settings.tokens)
    else:
        try:
            executor(code, settings.input, settings.implicit, settings.tokens)
        except Exception as err:
            print(err, file = sys.stderr)
