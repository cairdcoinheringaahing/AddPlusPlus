import functools
import inspect
import math
import operator
import random
import re
import sys

import error

CUSTOM_ERRORS = [m[0] for m in inspect.getmembers(error, inspect.isclass) if m[1].__module__ == 'error']

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
    def pop(self,index=-1):
        try: return super().pop(index)
        except: raise error.EmptyStackError()
    
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

    def __init__(self, code, args, stack=Stack(), line=0, general_code=''):
        self.args = args
        self.register = args[0] if args else 0
        self.stack = stack
        self.code = StackScript.tokenize(code)
        if self.code[-1] in 'BEb':
            self.code += ' '
        cont = False
        for i,cmd in enumerate(self.code):
            if cont:
                cont -= 1
                continue
            if cmd[0] == '"':
                self.stack.push(cmd[1:])
            elif isdigit(cmd):
                self.stack.push(eval_(cmd))
            else:
                if cmd == 'Q':
                    if self.stack.pop():
                        cont = -1
                    continue
                if cmd in 'BEb':
                    cmd += self.code[i+1]
                    cont = 1
                try:
                    result = self.COMMANDS[cmd]()
                except error.EmptyStackError:
                    raise error.EmptyStackError(line, general_code[line-1])
                except KeyError:
                    raise error.InvalidSymbolError(line, general_code[line-1], cmd)
                except Exception as n_error:
                    try:
                        raise n_error(line, general_code[line-1])
                    except Exception as e:
                        raise error.PythonError(line, ','.join(general_code[line-1]), e)
                if type(result) == Stack:
                    self.stack = result
                    del result

    @staticmethod
    def tokenize(text):
        final = []
        temp = ''
        num = ''
        instr = False
        
        for char in text:
            if char == '"': instr = not instr
		
            if instr:temp += char
            else:
                if temp:
                    final.append(temp)
                    temp = ''
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

        if temp: final.append(temp)
        if num: final.append(num)
        final = list(filter(lambda s: s!= '"', final))
        return final
    
    @property
    def COMMANDS(self):
        return {' ':lambda *_: None,

                '+':lambda: self.stack.push(add(self.stack.pop(), self.stack.pop())),
                '_':lambda: self.stack.push(subtract(self.stack.pop(), self.stack.pop())),
                '*':lambda: self.stack.push(multiply(self.stack.pop(), self.stack.pop())),
                '/':lambda: self.stack.push(divide(self.stack.pop(), self.stack.pop())),
                '^':lambda: self.stack.push(exponent(self.stack.pop(), self.stack.pop())),
                '%':lambda: self.stack.push(modulo(self.stack.pop(), self.stack.pop())),
                '@':lambda: self.stack.reverse(),
                '!':lambda: self.stack.push(not self.stack.pop()),
                '#':lambda: self.stack.sort(),
                ';':lambda: self.stack.push(self.stack.pop() * 2),
                '|':lambda: self.stack.push(abs(self.stack.pop())),
                '<':lambda: self.stack.push(self.stack.pop() < self.stack.pop()),
                '>':lambda: self.stack.push(self.stack.pop() > self.stack.pop()),
                '?':lambda: self.stack.push(bool(self.stack.pop())),
                '=':lambda: self.stack.push(self.stack.pop() == self.stack.pop()),
                'c':lambda: self.stack.clear(),
                'd':lambda: self.stack.push(self.stack[-1]),
                'D':lambda: self.stack.push(self.stack[-self.stack.pop()]),
                'L':lambda: self.stack.push(len(self.stack)),
                'P':lambda: self.stack.push(isprime(self.stack.pop())),
                'p':lambda: self.stack.pop(),
                'h':lambda: print(self.stack),
                'H':lambda: print(''.join(map(str, self.stack))),
                '&':lambda: self.stack.push(self.stack.pop() and self.stack.pop()),
                'S':lambda: self.stack.push(self.remove_duplicates()),
                's':lambda: self.stack.push(sum(self.stack)),
                'F':lambda: self.stack.push(*self.factors()),
                'f':lambda: self.stack.push(*filter(isprime, self.factors())),
                'A':lambda: self.stack.push(*self.args),
                'a':lambda: self.stack.push(list(self.args)),
                'N':lambda: self.stack.push('\n'.join(map(str, self.stack))),
                'O':lambda: self.stack.push(ord(self.stack.pop())),
                'C':lambda: self.stack.push(chr(self.stack.pop())),
                '$':lambda: self.stack.swap(),
                'o':lambda: self.stack.push(self.stack.pop() or self.stack.pop()),
                'M':lambda: self.stack.push(max(self.stack)),
                'm':lambda: self.stack.push(min(self.stack)),
                'n':lambda: self.join(),
                'R':lambda: self.stack.push(list(range(1, self.stack.pop()+1))),
                'r':lambda: self.stack.push(list(range(self.stack.pop(), self.stack.pop()))),
                'J':lambda: self.join(''),
                'j':lambda: self.join(str(self.stack.pop())),
                'V':lambda: self.store(self.stack.pop()),
                'G':lambda: self.stack.push(self.register),
		'x':lambda: self.stack.push([self.stack[-1] for _ in range(self.stack.pop())]),
		'i':lambda: self.stack.push(self.stack.pop() in self.stack.pop()),

                'Bx':lambda: self.stack.push(self.stack.pop() ^ self.stack.pop()),
                'Ba':lambda: self.stack.push(self.stack.pop() & self.stack.pop()),
                'Bo':lambda: self.stack.push(self.stack.pop() | self.stack.pop()),
                'Bf':lambda: self.stack.push(~self.stack.pop()),
                'BB':lambda: self.stack.push(bin(self.stack.pop())[2:]),
                'Bb':lambda: self.stack.push(int(self.stack.pop(), self.stack.pop())),
                'Bc':lambda: self.columns(),
                'B+':lambda: self.apply(lambda l: functools.reduce(operator.add, l)),
                'B*':lambda: self.apply(lambda l: functools.reduce(operator.mul, l)),
                'B\\':lambda: self.apply(lambda l: functools.reduce(operator.floordiv, l)),
                'B_':lambda: self.apply(lambda l: functools.reduce(operator.sub, l)),
                'B`':lambda: self.apply(lambda l: functools.reduce(operator.pow, l)),
                'B%':lambda: self.apply(lambda l: functools.reduce(operator.mod, l)),
                'B/':lambda: self.apply(lambda l: functools.reduce(operator.truediv, l)),
                'B&':lambda: self.apply(lambda l: functools.reduce(operator.and_, l)),
                'B|':lambda: self.apply(lambda l: functools.reduce(operator.or_, l)),
                'B^':lambda: self.apply(lambda l: functools.reduce(operator.xor, l)),
                'B=':lambda: self.apply(lambda l: self.eq(*l)),
                'B!':lambda: self.apply(operator.not_),
                'B~':lambda: self.apply(operator.inv),
                'BM':lambda: self.apply(max),
                'Bm':lambda: self.apply(min),
                'B]':lambda: self.wrap(),
                'BC':lambda: self.stack.push(int(''.join(map(str, self.stack.pop())), self.stack.pop())),
                'BR':lambda: self.stack.push(self.stack.pop()[::-1]),
                'BF':lambda: self.flatten(),
				
		'E#':lambda: Stack([sorted(i) for i in self.stack]),
		'E@':lambda: Stack([i[::-1] for i in self.stack]),
		'ER':lambda: Stack([list(range(1, i+1)) for i in self.stack]),
		'EC':lambda: self.collect(),
		'ED':lambda: Stack([[int(i) for i in list(str(j))] for j in self.stack]),
		'Ed':lambda: Stack([int(''.join(map(str, i))) for i in self.stack]),
		'Ep':lambda: Stack([i[:-1] for i in self.stack]),
		'EP':lambda: Stack([i[1:] for i in self.stack]),
		'EL':lambda: Stack([len(i) for i in self.stack]),
		'Es':lambda: Stack([sum(i) for i in self.stack]),
		'E|':lambda: Stack([abs(i) for i in self.stack]),
		'E_':lambda: Stack([-i for i in self.stack]),
		'EQ':lambda: Stack([self.remove_duplicates(i) for i in self.stack]),
		'Ei':lambda: self.stack.push([i in self.stack[-1] for i in self.stack.pop()]),
		'EZ':lambda: Stack(filter(None, self.stack)),
		'EF':lambda: Stack([i for i in self.stack[:-1] if i not in self.stack[-1]]),
		'Ef': lambda: Stack([i for i in self.stack[:-1] if i in self.stack[-1]]),

                'bM':lambda: self.stack.push(max(self.stack.pop())),
                'bm':lambda: self.stack.push(min(self.stack.pop())),
                'b]':lambda: self.stack.push([self.stack.pop()]),
                'b+':lambda: self.stack.push(functools.reduce(operator.add, self.stack.pop())),
                'b_':lambda: self.stack.push(functools.reduce(operator.sub, self.stack.pop())),
                'b*':lambda: self.stack.push(functools.reduce(operator.mul, self.stack.pop())),
                'b\\':lambda: self.stack.push(functools.reduce(operator.floordiv, self.stack.pop())),
                'b`':lambda: self.stack.push(functools.reduce(operator.pow, self.stack.pop())),
                'b%':lambda: self.stack.push(functools.reduce(operator.mod, self.stack.pop())),
                'b/':lambda: self.stack.push(functools.reduce(operator.truediv, self.stack.pop())),
                'b&':lambda: self.stack.push(functools.reduce(operator.and_, self.stack.pop())),
                'b|':lambda: self.stack.push(functools.reduce(operator.or_, self.stack.pop())),
                'b^':lambda: self.stack.push(functools.reduce(operator.xor, self.stack.pop())),
                'b=':lambda: self.stack.push(self.eq(*self.stack.pop())),
                'b!':lambda: self.stack.push(list(map(operator.not_, self.stack.pop()))),
                'b~':lambda: self.stack.push(list(map(operator.inv, self.stack.pop()))),
                'bB':lambda: self.pad_bin(),
                'bU':lambda: self.stack.push(*self.stack.pop()),
                'bF':lambda: self.stack.push(self.flatten(self.stack.pop())),
               }

    def apply(self, func):
        self.stack = Stack(map(func, self.stack))
        
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
        self.stack = Stack(array)
		
    def columns(self):
        self.stack = Stack(map(list, zip(*self.stack)))
        
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
        
    def join(self, char='\n'):
        newstack = Stack()
        newstack.push(char.join(map(str, self.stack)))
        self.stack = newstack
		
    def pad_bin(self):
        copy = self.stack.copy()
        length = max(map(lambda a: len(bin(a)[2:]), copy))
        for i in range(len(self.stack)):
            self.stack[i] = Stack(map(eval_, bin(self.stack[i])[2:].rjust(length, '0')))
		
    def remove(self, even_odd):
        self.stack = Stack(filter(lambda x: x%2 == int(bool(even_odd)), self.stack))
        
    def remove_duplicates(self, array=None):
        final = []
        if array is None: array = self.stack
        for s in array:
            if s not in final:
                final.append(s)
        return final
	
    def run(self,flag,text):
        if flag:
            return self.stack
        if text:
            return ''.join(list(map(StackScript.stringify, self.stack)))
        return self.stack.pop()
        
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

    def __init__(self, name, args, code, line, g_code, *flags):
        self.name = name
        self.args = args
        self.code = code
        self.stack = Stack()
        self.flags = list(flags)
        self.line = line
        self.gen = g_code

    def __call__(self, *args):
        if not self.flags[2]:
            args = list(args)[:self.args]
            while len(args) != self.args:
                args.append(-1)
                
        if self.flags[4]:
            self.stack.push(list(args))
        else:
            self.stack.push(*args)
        script = StackScript(self.code, args, self.stack, self.line, self.gen)
        value = script.run(*self.flags[:2])
        self.stack = Stack()
        
        if self.flags[3]:
            print(value)
            return Null(value)
        return int(value) if type(value) == bool else value
        
    def __repr__(self):
        return '<Function ${}: {}>'.format(self.name, self.code)

class Script:

    def __init__(self,code, inputs, _, __):

        self.NILADS = r'!~&#@NPOHSQVG'
        self.MONADS = r'+-*/\^><%=R'
        self.CONSTRUCTS = 'FWEIDL'
        
        self.code = list(filter(None, code.split('\n')))
            
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
                    func_code = ','.join(cmd[3:])+' '
                    self.functions[func_name] = Function(func_name, func_args, func_code, self.line, code, *func_flags)
                elif cmd[0] == 'L':
                    cmd = cmd.split(',')
                    flags = cmd[0][1:]
                    lambda_c = ','.join(cmd[1:])
                    lambda_n = len(list(filter(lambda a: bool(re.search(r'^lambda \d+$', a)), self.functions.keys()))) + 1
                    name = 'lambda {}'.format(lambda_n)
                    lambda_f = []
                    for flag in '*^?:!':
                        lambda_f.append(flag == '?' or flag in flags)
                    self.functions[name] = Function(name, 0, lambda_c, self.line, code, *lambda_f)
                    
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
