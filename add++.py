import functools
import math
import operator
import random
import re
import sys

import error

def isdigit(string):
    return all(i in '1234567890-.' for i in string)

def eval_(string):
    if '.' in string: return float(string)
    try: return int(string)
    except: return string

class Stack(list):
    def push(self, *values):
        for v in values:
            try: self.append(v.replace("'",'"'))
            except: self.append(v)
    def pop(self, index=-1):
        return super().pop(index)
    def peek(self, index=-1):
        return self[index]
    
    def swap(self):
        self[-1], self[-2] = self[-2], self[-1]
        
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
            return False
    return x > 1 and isinstance(x, int)

class Null:
    def __init__(self, value):
        self.value = value

class StackScript:

    def __init__(self, code, args, funcs, stack, line, outer):
        self.args = args
        self.register = args if args else 0
        self.stacks = [stack]
        self.index = 0
        self.code = StackScript.tokenize(code + ' ')
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
                    result = self.COMMANDS[cmd]()
                except TypeError:
                    raise error.IncongruentTypesError(line, outer[line-1], cmd)
                except:
                    raise error.EmptyStackError(line, outer[line-1])
                    
                if result == Null:
                    raise error.InvalidSymbolError(line, outer[line-1], cmd)
                
                if type(result) == Stack:
                    self.stacks[self.index] = result
                    del result

    @staticmethod
    def tokenize(text):
        final = []
        stemp = ''
        ctemp = ''
        num = ''
        instr = False
        incall = False
        text = text.replace('{', ' {')
        
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
        
        return tokens
    
    @property
    def COMMANDS(self):
        return {
                '!': lambda: self.stack.push(not self.stack.pop()),
                '#': lambda: self.stack.sort(),
                '$': lambda: self.stack.swap(),
                '%': lambda: self.stack.push(modulo(self.stack.pop(), self.stack.pop())),
                '&': lambda: self.stack.push(self.stack.pop() and self.stack.pop()),
                "'": lambda: self.stack.push(self.stack.pop() * 2),
                '(': lambda: self.decrement(),
                ')': lambda: self.increment(),
                '*': lambda: self.stack.push(multiply(self.stack.pop(), self.stack.pop())),
                '+': lambda: self.stack.push(add(self.stack.pop(), self.stack.pop())),
                '/': lambda: self.stack.push(divide(self.stack.pop(), self.stack.pop())),
                ':': lambda: Null,
                '<': lambda: self.stack.push(self.stack.pop() < self.stack.pop()),
                '=': lambda: self.stack.push(self.stack.pop() == self.stack.pop()),
                '>': lambda: self.stack.push(self.stack.pop() > self.stack.pop()),
                '?': lambda: self.stack.push(bool(self.stack.pop())),
                '@': lambda: self.stack.reverse(),
                
                'A': lambda: self.stack.push(*self.args),
                'B': lambda: self.stack.push(self.stack[:self.stack.pop()]),
                'C': lambda: self.stack.push(chr(self.stack.pop())),
                'D': lambda: self.stack.push(self.stack[-self.stack.pop()]),
                'E': lambda: self.stack.push(list(map(list, enumerate(self.stack.pop())))),
                'F': lambda: self.stack.push(*self.factors()),
                'G': lambda: self.stack.push(self.register),
                'H': lambda: print(''.join(map(str, self.stack))),
                'I': lambda: Null,
                'J': lambda: self.join(''),
                'K': lambda: Null,
                'L': lambda: self.stack.push(len(self.stack)),
                'M': lambda: self.stack.push(max(self.stack)),
                'N': lambda: self.stack.push('\n'.join(map(str, self.stack))),
                'O': lambda: self.stack.push(ord(self.stack.pop())),
                'P': lambda: self.stack.push(isprime(self.stack.pop())),
                'R': lambda: self.stack.push(list(range(1, self.stack.pop()+1))),
                'S': lambda: self.stack.push(self.remove_duplicates()),
                'T': lambda: Null,
                'U': lambda: Null,
                'V': lambda: self.store(self.stack.pop()),
                'X': lambda: self.stack.push([[self.stack[-1] for _ in range(self.stack.pop())], self.stack.pop()][0]),
                'Y': lambda: Null,
                'Z': lambda: Null,

                '[': lambda: self.stack.push(self.prevcall),
                ']': lambda: self.run_lambda(self.stack.pop()),
                '^': lambda: self.stack.push(exponent(self.stack.pop(), self.stack.pop())),
                '_': lambda: self.stack.push(subtract(self.stack.pop(), self.stack.pop())),
                '`': lambda: Null,
                
                'a': lambda: self.stack.push(list(self.args)),
                'b': lambda: Null,
                'c': lambda: self.stack.clear(),
                'd': lambda: self.stack.push(self.stack[-1]),
                'e': lambda: self.stack.push(self.stack.pop() in self.stack.pop()),
                'f': lambda: self.stack.push(*filter(isprime, self.factors())),
                'g': lambda: Null,
                'h': lambda: print(self.stack),
                'i': lambda: self.stack.push(int(self.stack.pop())),
                'j': lambda: self.join(str(self.stack.pop())),
                'k': lambda: Null,
                'l': lambda: Null,
                'm': lambda: self.stack.push(min(self.stack)),
                'n': lambda: self.join(),
                'o': lambda: self.stack.push(self.stack.pop() or self.stack.pop()),
                'p': lambda: self.stack.pop(),
                'q': lambda: self.stack.push(set(self.stack.pop())),
                'r': lambda: self.stack.push(list(range(self.stack.pop(), self.stack.pop()))),
                's': lambda: self.stack.push(sum(self.stack)),
                't': lambda: Null,
                'u': lambda: Null,
                'v': lambda: self.stack.push(eval(self.stack.pop())),
                'w': lambda: Null,
		'x': lambda: self.stack.push([self.stack[-1] for _ in range(self.stack.pop())]),
		'y': lambda: [self.stack.push(self.stack[-1]) for _ in range(self.stack.pop())],
                'z': lambda: Null,
                
                '|': lambda: self.stack.push(abs(self.stack.pop())),
                '~': lambda: Null,

                'B!':lambda: self.apply(operator.not_),
                'B#':lambda: self.apply(sorted),
                'B$':lambda: Null,
                'B%':lambda: self.apply(lambda l: functools.reduce(operator.mod, l)),
                'B&':lambda: self.apply(lambda l: functools.reduce(operator.and_, l)),
                "B'":lambda: self.apply(lambda x: 2 * x),
                'B(':lambda: self.stack.push(self.stacks[self.index - 1].pop()),
                'B)':lambda: self.stack.push(self.stacks[(self.index + 1) % len(self.stacks)].pop()),
                'B*':lambda: self.apply(lambda l: functools.reduce(operator.mul, l)),
                'B+':lambda: self.apply(lambda l: functools.reduce(operator.add, l)),
                'B/':lambda: self.apply(lambda l: functools.reduce(operator.truediv, l)),
                'B:':lambda: Null,
                'B<':lambda: Null,
                'B=':lambda: self.apply(lambda l: self.eq(*l)),
                'B>':lambda: Null,
                'B?':lambda: Null,
                'B@':lambda: self.apply(reversed, True),
                
                'BA':lambda: self.apply(abs),
                'BB':lambda: self.stack.push(bin(self.stack.pop())[2:]),
                'BC':lambda: self.collect(),
                'BD':lambda: self.apply(lambda i: list(map(int, str(i)))),
                'BE':lambda: self.apply(lambda i: i in self.stack[-1]),
                'BF':lambda: self.flatten(),
                'BG':lambda: Null,
                'BH':lambda: Null,
                'BI':lambda: Null,
                'BJ':lambda: self.apply(lambda i: ''.join(map(str, i))),
                'BK':lambda: Null,
                'BL':lambda: self.apply(len),
                'BM':lambda: self.apply(max),
                'BN':lambda: Null,
                'BO':lambda: Null,
                'BP':lambda: self.apply(lambda x: x[1:]),
                'BQ':lambda: self.apply(self.remove_duplicates),
                'BR':lambda: self.apply(lambda x: list(range(1, x + 1))),
                'BS':lambda: Stack([self.stack[i : i+2] for i in range(len(self.stack) - 1)]),
                'BT':lambda: Null,
                'BU':lambda: Null,
                'BV':lambda: self.stack.push(exec(self.stack.pop())),
                'BW':lambda: Stack([i for i in self.stack[:-1] if i not in self.stack[-1]]),
                'BX':lambda: self.stack.push(random.choice(self.stack.pop())),
                'BY':lambda: self.apply(random.choice),
                'BZ':lambda: Stack(filter(None, self.stack)),

                'B]':lambda: self.wrap(),
                'B[':lambda: self.apply(lambda l: [l]),
                'B^':lambda: self.apply(lambda l: functools.reduce(operator.xor, l)),
                'B_':lambda: self.apply(lambda l: functools.reduce(operator.sub, l)),
                'B`':lambda: self.apply(lambda l: functools.reduce(operator.pow, l)),

                'Ba':lambda: self.stack.push(self.stack.pop() & self.stack.pop()),
                'Bb':lambda: self.stack.push(int(self.stack.pop(), self.stack.pop())),
                'Bc':lambda: self.columns(),
                'Bd':lambda: self.apply(lambda l: functools.reduce(operator.floordiv, l)),
                'Be':lambda: self.stack.push([i in self.stack[-1] for i in self.stack.pop()]),
                'Bf':lambda: self.stack.push(~self.stack.pop()),
                'Bg':lambda: Null,
                'Bh':lambda: Null,
                'Bi':lambda: self.apply(int),
                'Bj':lambda: Null,
                'Bk':lambda: Null,
                'Bl':lambda: Null,
                'Bm':lambda: self.apply(min),
                'Bn':lambda: self.apply(lambda i: -i),
                'Bo':lambda: self.stack.push(self.stack.pop() | self.stack.pop()),
                'Bp':lambda: self.apply(lambda x: x[:-1]),
                'Bq':lambda: Null,
                'Br':lambda: Null,
                'Bs':lambda: self.apply(sum),
                'Bt':lambda: Null,
                'Bu':lambda: Null,
                'Bv':lambda: self.apply(lambda i: int(''.join(map(str, i)))),
                'Bw':lambda: Stack([i for i in self.stack[:-1] if i in self.stack[-1]]),
                'Bx':lambda: self.stack.push(self.stack.pop() ^ self.stack.pop()),
                'By':lambda: Null,
                'Bz':lambda: Null,
                
                'B|':lambda: self.apply(lambda l: functools.reduce(operator.or_, l)),
                'B~':lambda: self.apply(operator.inv),
                
                'b!':lambda: self.stack.push(list(map(operator.not_, self.stack.pop()))),
                'b#':lambda: Null,
                'b$':lambda: Null,
                'b%':lambda: self.stack.push(functools.reduce(operator.mod, self.stack.pop())),
                'b&':lambda: self.stack.push(functools.reduce(operator.and_, self.stack.pop())),
                "b'":lambda: self.stack.push([i * 2 for i in self.stack.pop()]),
                'b(':lambda: self.stacks[self.index - 1].push(self.stack.pop()),
                'b)':lambda: self.stacks[(self.index + 1) % len(self.stacks)].push(self.stack.pop()),
                'b*':lambda: self.stack.push(functools.reduce(operator.mul, self.stack.pop())),
                'b+':lambda: self.stack.push(functools.reduce(operator.add, self.stack.pop())),
                'b/':lambda: self.stack.push(functools.reduce(operator.truediv, self.stack.pop())),
                'b:':lambda: Null,
                'b<':lambda: Null,
                'b=':lambda: self.stack.push(self.eq(*self.stack.pop())),
                'b>':lambda: Null,
                'b?':lambda: Null,
                'b@':lambda: Null,

                'bA':lambda: Null,
                'bB':lambda: Null,
                'bC':lambda: Null,
                'bD':lambda: Null,
                'bE':lambda: Null,
                'bF':lambda: self.stack.push(self.flatten(self.stack.pop())),
                'bG':lambda: Null,
                'bH':lambda: Null,
                'bI':lambda: Null,
                'bJ':lambda: Null,
                'bK':lambda: Null,
                'bL':lambda: self.stack.push(len(self.stack.pop())),
                'bM':lambda: self.stack.push(max(self.stack.pop())),
                'bN':lambda: Null,
                'bO':lambda: Null,
                'bP':lambda: Null,
                'bQ':lambda: Null,
                'bR':lambda: self.stack.push(self.stack.pop()[::-1]),
                'bS':lambda: Null,
                'bT':lambda: Null,
                'bU':lambda: self.stack.push(*self.stack.pop()),
                'bV':lambda: Null,
                'bW':lambda: Null,
                'bX':lambda: Null,
                'bY':lambda: Null,
                'bZ':lambda: Null,

                'b[':lambda: Null,
                'b]':lambda: self.stack.push([self.stack.pop()]),
                'b^':lambda: self.stack.push(functools.reduce(operator.xor, self.stack.pop())),
                'b_':lambda: self.stack.push(functools.reduce(operator.sub, self.stack.pop())),
                'b`':lambda: self.stack.push(functools.reduce(operator.pow, self.stack.pop())),

                'ba':lambda: Null,
                'bb':lambda: Null,
                'bc':lambda: Null,
                'bd':lambda: self.stack.push(functools.reduce(operator.floordiv, self.stack.pop())),
                'be':lambda: Null,
                'bf':lambda: Null,
                'bg':lambda: Null,
                'bh':lambda: Null,
                'bi':lambda: Null,
                'bj':lambda: Null,
                'bk':lambda: Null,
                'bl':lambda: Null,
                'bm':lambda: self.stack.push(min(self.stack.pop())),
                'bn':lambda: Null,
                'bo':lambda: Null,
                'bp':lambda: Null,
                'bq':lambda: Null,
                'br':lambda: Null,
                'bs':lambda: Null,
                'bt':lambda: Null,
                'bu':lambda: Null,
                'bv':lambda: Null,
                'bw':lambda: Null,
                'bx':lambda: Null,
                'by':lambda: Null,
                'bz':lambda: Null,
                
                'b|':lambda: self.stack.push(functools.reduce(operator.or_, self.stack.pop())),
                'b~':lambda: self.stack.push(list(map(operator.inv, self.stack.pop()))),
                
                

                
               }

    def apply(self, func, array = False):
        if array:
            return Stack(map(lambda v: list(func(v)), self.stack))
        return Stack(map(func, self.stack))
        
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
       def flatten_array(array):
           flat = []
           if type(array) == list:
               for item in array:
                   flat += flatten_array(item)
           else:
               flat.append(array)
           return flat

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
        return ret.pop()

    def run_lambda(self, index):
        lamb = self.functions['lambda {}'.format(index)]
        self.prevcall = lamb(*self.stack)
        self.stack.clear()
        self.stack.push(self.prevcall)
        
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

    def __init__(self, name, args, code, line, g_code, outerf, *flags):
        self.name = name
        self.args = args if args != -1 else 0
        self.lamb = args == -1
        self.code = code
        self.stack = Stack()
        self.flags = list(flags)
        self.line = line
        self.gen = g_code
        self.outerf = outerf

    def __call__(self, *args, funccall = False):
        if not self.flags[2]:
            args = list(args)[:self.args]
            while len(args) != self.args:
                args.append(-1)
                
        if self.flags[4]:
            self.stack.push(list(args))
        else:
            self.stack.push(*args)
        script = StackScript(self.code, args, self.outerf, self.stack, self.line, self.gen)
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

    def __init__(self,code, inputs=()):

        self.NILADS = r'!~&#@NPOHSQVG'
        self.MONADS = r'+-*/\^><%=R'
        self.CONSTRUCTS = 'FWEIDL'
        
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
                    for flag in '*^?:!':
                        func_flags.append(flag in cmd[2])
                    func_code = ','.join(cmd[3:])
                    self.functions[func_name] = Function(func_name, func_args, func_code,
                                                         self.line, code, self.functions, *func_flags)
                elif cmd[0] == 'L':
                    cmd = cmd.split(',')
                    flags = cmd[0][1:]
                    lambda_c = ','.join(cmd[1:])
                    lambda_n = len(list(filter(lambda a: bool(re.search(r'^lambda \d+$', a)), self.functions.keys()))) + 1
                    name = 'lambda {}'.format(lambda_n)
                    lambda_f = []
                    for flag in '*^?:!':
                        lambda_f.append(flag == '?' or flag in flags)
                    self.functions[name] = Function(name, -1, lambda_c, self.line,
                                                    code, self.functions, *lambda_f)
                    
            else:
                self.implicit = True
                if cmd[:2] in ['x:', 'y:']:
                    if cmd[0] == 'x': acc = self.x; acc_n = 'x'
                    else: acc = self.y; acc_n = 'y'
                        
                    c = cmd[2:]
                    if c == '?':
                        try: acc = inputs[self.I]; self.I += 1
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
                            try: args.append(inputs[self.I]); self.I += 1
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
                    
        if not self.called and self.functions:
            func = self.functions[list(self.functions.keys())[0]]
            if self.I < len(inputs): result = func(*inputs[self.I:])
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
            for i in inputs:
                self.stored.append(i)
        if symbol == '}': self.x, self.y = self.y, self.x
            
        if len(cmd) > 1: value = eval_(cmd[1:])
        else: value = None

        if cmd[:2] in ['x:', 'y:']:
            if cmd[0] == 'x': acc = self.x; acc_n = 'x'
            else: acc = self.y; acc_n = 'y'
                
            c = cmd[2:]
            if c == '?':
                try: acc = inputs[self.I]; self.I += 1
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
                    try: args.append(inputs[self.I]); self.I += 1
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
                try: value = inputs[self.I];  self.I += 1
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
                    try: v = inputs[self.I]; self.I += 1
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
        
    def _print(self):
        if self.string: print(end=self.string)
        else: print(end=chr(self.x))
            
    def print(self):
        if self.string: print(self.string)
        else: print(chr(self.x))
            
    def randint(self,y=0):
        if y > self.x: return random.randint(self.x, y)
        return random.randint(y, self.x)

if __name__ == '__main__':

    program = sys.argv[1]
    inputs = list(map(eval_, sys.argv[2:]))

    if '--error' in sys.argv[2:]:
        if program.endswith(('.txt', '.app')):
            Script(open(program).read(), inputs)
        else:
            Script(program, inputs)
    else:
        try:
            if program.endswith(('.txt', '.app')):
                Script(open(program).read(), inputs)
            else:
                Script(program, inputs)
        except Exception as e:
            print(e, file=sys.stderr)
