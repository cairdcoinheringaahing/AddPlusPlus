import functools
import math
import operator
import random
import sys

class EmptyStackError(Exception):
    def __init__(self, num, line):
        
        self.message = '''Fatal error: EmptyStackError
    line {}: '{}'
        The stack is empty, and is unable to be popped from'''.format(num, line)
        
        super(EmptyStackError, self).__init__(self.message)

class UnableToRetrieveFunctionError(Exception):
    def __init__(self, num, line, name):

        self.message = '''Fatal error: UnableToRetrieveFunctionError
    line {}: '{}'
        Unable to retrieve function {}. {} must be defined before calling'''.format(num, line, name, name)

        super(UnableToRetrieveFunctionError, self).__init__(self.message)

class EmptySecondStackError(Exception):
    def __init__(self, num, line):
        
        self.message = '''Fatal error: EmptySecondStackError
    line {}: '{}'
        The second stack is empty, and is unable to be popped from'''.format(num, line)

        super(EmptySecondStackError, self).__init__(self.message)

class NoMoreInputError(Exception):
    def __init__(self, num, line):
        
        self.message = '''Fatal error: NoMoreInputError
    line {}: '{}'
        All input has been used, and cannot be used again'''.format(num, line)

        super(NoMoreInputError, self).__init__(self.message)

class InvalidSymbolError(Exception):
    def __init__(self, num, line, char):
        
        self.message = '''Fatal error: InvalidSymbolError
    line {}: '{}'
        The character '{}' is an invalid Add++ character'''.format(num, line, char)

        super(InvalidSymbolError, self).__init__(self.message)

class DivisionByZeroError(Exception):
    def __init__(self, num, line):
        
        self.message = '''Fatal error: DivisionByZeroError
    line {}: '{}'
        The laws of mathematics dictate that you are unable to divide by 0'''.format(num, line)

        super(DivisionByZeroError, self).__init__(self.message)

class PythonError(Exception):
    def __init__(self, num, line, error):

        self.message = '''Fatal error: PythonError
    line {}: '{}'
        Python error raised: {}'''.format(num, line, error)

        super(PythonError, self).__init__(self.message)

CUSTOM_ERRORS = [EmptyStackError, UnableToRetrieveFunctionError, EmptySecondStackError, NoMoreInputError, InvalidSymbolError, DivisionByZeroError]

def isdigit(string):
    return all(i in '1234567890-.' for i in string)

def eval_(string):
    if '.' in string:
        return float(string)
    try:
        return int(string)
    except:
        return string

class Stack(list):
    def push(self, *values):
        for v in values:
            try:
                self.append(v.replace("'",'"'))
            except:
                self.append(v)
    def pop(self,index=-1):
        try:
            return super().pop(index)
        except:
            raise EmptyStackError()
    
    def swap(self):
        self[-1], self[-2] = self[-2], self[-1]
        
def add(x,y):
    if type(x) == str and type(y) == int:
        return ''.join(map(lambda c: chr(ord(c)+y), x))
    if type(x) == int and type(y) == str:
        return ''.join(map(lambda c: chr(ord(c)+x), y))
    return x + y

def subtract(x,y):
    if type(x) == str and type(y) == str:
        for char in y:
            x = x.replace(char, "")
        return x
    if type(x) == str and type(y) == int:
        return ''.join(map(lambda c: chr(ord(c)-y), x))
    if type(x) == int and type(y) == str:
        return ''.join(map(lambda c: chr(ord(c)-x), y))
    return x - y

def multiply(x,y):
    if type(x) == str and type(y) == str:
        final = ""
        for a,b in zip(x,y):
            final += chr(ord(a)+ord(b))
        return final
    if type(x) == str and type(y) == int:
        return ''.join(map(lambda c: c*y, x))
    if type(x) == int and type(y) == str:
        return x * y
    return x * y

def divide(x,y):
    if type(x) == str and type(y) == str:
            final = ""
            for a,b in zip(x,y):
                final += chr(max(ord(a),ord(b))-min(ord(a),ord(b)))
            return final
    if type(x) == str and type(y) == int:
        return ''.join(map(lambda c: chr(ord(c)//y), x))
    if type(x) == int and type(y) == str:
        return ''.join(map(lambda c: chr(ord(c)//x), y))
    return x / y

def exponent(x,y):
    if type(x) == str and type(y) == str:
        final = ""
        for a,b in zip(x,y):
            final += [a,b][b>a]
        return final
    if type(x) == int and type(y) == int:
        return x ** y
    c = x if type(x) == int else y
    d = y if c == x else x
    final = d
    for i in range(c-1):
        final = multiply(final,d)
    return final

def modulo(x,y):
    if type(x) == str and type(y) == str:
        return x % y
    if type(x) == str and type(y) == int:
        return ''.join(map(lambda c: chr(ord(c)%y), x))
    if type(x) == int and type(y) == str:
        return ''.join(map(lambda c: chr(ord(c)%x), y))
    return x%y

def absolute(x):
    if type(x) == str:
        return x
    return abs(x)

def isprime(x):
    if type(x) == str:
        return False
    for i in range(2,x):
        if x%i == 0:
            return False
    return True

class Null:
    def __init__(self, value):
        self.value = value

class StackScript:

    def __init__(self, code, args, stack=Stack(), line=0, general_code=''):
        self.args = args
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
                except EmptyStackError:
                    raise EmptyStackError(line, ','.join(general_code[line-1]))
                except KeyError:
                    raise InvalidSymbolError(line, ','.join(general_code[line-1]), cmd)
                except Exception as error:
                    try:
                        raise error(line, ','.join(general_code[line-1]))
                    except Exception as e:
                        raise PythonError(line, ','.join(general_code[line-1]), e)
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
            
            if char == '"':
                instr = not instr

            if instr:
                temp += char
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
                    except:
                        final.append(char)
                else:
                    if num:
                        final.append(num)
                        num = ''
                    final.append(char)

        if temp:
            final.append(temp)
        if num:
            final.append(num)

        final = list(filter(lambda s: s!= '"', final))
        return final
    
    @property
    def COMMANDS(self):
        return {' ':lambda *a: None,

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
                '|':lambda: self.stack.push(absolute(self.stack.pop())),
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
                'S':lambda: self.stack.push(*self.remove_duplicates()),
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
        
    def eq(self, *args):
        incs = [args[i] == args[i-1] for i in range(1, len(args))]
        return all(incs)
        
    def join(self, char='\n'):
        newstack = Stack()
        newstack.push(char.join(map(str, self.stack)))
        self.stack = newstack
    
    def factors(self):
        lof = []
        x = self.stack.pop()
        if type(x) == str:
            return list(x)
        for i in range(1,int(x)):
            if x%i == 0:
                lof.append(i)
        return lof
    
    def remove_duplicates(self):
        final = []
        for s in self.stack:
            if s not in final:
                final.append(s)
        return final
    
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

    @staticmethod
    def stringify(value):
        try:
            return chr(int(abs(value)))
        except:
            return str(value)

    def run(self,flag,text):
        if flag:
            return self.stack
        if text:
            return ''.join(list(map(StackScript.stringify, self.stack)))
        return self.stack.pop()
    
    def remove(self,even_odd):
        self.stack = Stack(filter(lambda x: x%2 == int(bool(even_odd)), self.stack))

    def pad_bin(self):
        copy = self.stack.copy()
        length = max(map(lambda a: len(bin(a)[2:]), copy))
        for i in range(len(self.stack)):
            self.stack[i] = Stack(map(eval_, bin(self.stack[i])[2:].rjust(length, '0')))

    def columns(self):
        self.stack = Stack(map(list, zip(*self.stack)))

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
            self.stack.push(args)
        else:
            self.stack.push(*args)
        script = StackScript(self.code, args, self.stack, self.line, self.gen)
        value = script.run(*self.flags[:2])
        
        if self.flags[3]:
            print(value)
            return Null(value)
        return value
        
    def __repr__(self):
        return '<Function ${}: {}>'.format(self.name, self.code)

class Script:

    def __init__(self,code,inputs,_=0,__=0,x=0,recur=False):

        self.NILADS = r'!~&#@NPOHSQVG'
        self.MONADS = r'+-*/\^><%=R'
        
        code = list(map(lambda x: x.split(','), filter(None, code.split('\n'))))
            
        if not recur:
            self.called = False
            self.implicit = False
            self.stored = []
            self.string = ''
            self.functions = {}
            I = 0
            self.y = 0
            line = 0

        self.x = x
        
        f = code[:]
        code.clear()
        for i in range(len(f)):
            if len(f[i]) > 1:
                code.append(f[i])
            else:
                code.append(f[i][0])
        
        try:
            if code[0] not in 'FIEWD':
                self.x = x
                self.code = code
        except:
            if code[0][0] not in 'FEIWD':
                self.x = x
                self.code = code
            
        for cmd in code:
            
            if not recur:
                line += 1
                
            if type(cmd) == list:
                if cmd[0] == 'F':
                    for i in range(int(self.x)):
                        self.__init__('\n'.join(cmd),inputs,recur=True)
                if cmd[0] == 'E':
                    for i in range(int(self.x)):
                        self.__init__('\n'.join(cmd),inputs,x=i,recur=True)
                if cmd[0] == 'I':
                    if self.x:
                        self.__init__('\n'.join(cmd),inputs,recur=True)
                if cmd[0] == 'W':
                    while self.x:
                        self.__init__('\n'.join(cmd),inputs,x=self.x,recur=True)
                if cmd[0] == 'D':
                    func_name = cmd[1]
                    func_args = cmd[2].count('@')
                    func_flags = []
                    for flag in '*^?:!':
                        func_flags.append(flag in cmd[2])
                    func_code = ','.join(cmd[3:])+' '
                    self.functions[func_name] = Function(func_name, func_args, func_code, line, code, *func_flags)
                    
            else:
                self.implicit = True
                if cmd[:2] == 'x:':
                    c = cmd[2:]
                    if c == '?':
                        try:
                            self.x = inputs[I]
                            I += 1
                        except:
                            self.x = 0
                    elif c == 'x':
                        self.x = self.x
                    elif c == 'y':
                        self.x = self.y
                    elif c == 'G':
                        self.x = self.stored.pop()
                    elif c == 'g':
                        self.x = self.stored[-1]
                    else:
                        self.x = eval_(c)
                        
                elif cmd[:2] == 'y:':
                    c = cmd[2:]
                    if c == '?':
                        try:
                            self.y = inputs[I]
                            I += 1
                        except:
                            self.y = 0
                    elif c == 'x':
                        self.y = self.x
                    elif c == 'y':
                        self.y = self.y
                    elif c == 'G':
                        self.y = self.stored.pop()
                    elif c == 'g':
                        self.y = self.stored[-1]
                    else:
                        self.y = eval_(c)
                        
                elif cmd[0] == '$':
                    self.called = True
                    cmd = cmd.split('>')
                    try:
                        func = self.functions[cmd[0][1:]]
                    except:
                        raise UnableToRetrieveFunctionError(line, code[line-1], cmd[0][1:])
                    args = []
                    for c in cmd[1:]:
                        if c == '?':
                            try:
                                args.append(inputs[I])
                                I += 1
                            except:
                                args.append(0)
                        elif c == 'x':
                            args.append(self.x)
                        elif c == 'y':
                            args.append(self.y)
                        elif c == 'G':
                            args.append(self.stored.pop())
                        elif c == 'g':
                            args.append(self.stored[-1])
                        elif c == '_':
                            for element in self.stored:
                                args.append(element)
                        else:
                            args.append(eval_(c))
                            
                    value = func(*args)
                    if type(value) == Null:
                        value = value.value
                    if type(value) == list:
                        for v in value:
                            self.stored.append(v)
                    if type(value) == str:
                        self.stored.append(value)
                    self.x = value
                else:
                    
                    symbol = cmd[0]
                    if symbol == "_":
                        for i in inputs:
                            self.stored.append(i)
                    if symbol == '}':
                        self.x, self.y = self.y, self.x
                    if len(cmd) > 1:
                        value = eval_(cmd[1:])
                    else:
                        value = None

                    if value is not None:
                        if value == '?':
                            try:
                                value = inputs[I]
                                I += 1
                            except:
                                raise NoMoreInputError(line, code[line-1])
                        if value == 'G':
                            try:
                                value = self.stored.pop()
                            except:
                                raise EmptySecondStackError(line, code[line-1])
                        if value == 'g':
                            value = self.stored[-1]
                        if value == 'x':
                            value = self.x
                        if value == 'y':
                            value = self.y
                        try:
                            self.x = self.COMMANDS[symbol](value)
                        except ZeroDivisionError:
                            raise DivisionByZeroError(line, code[line-1])
                        except:
                            raise InvalidSymbolError(line, code[line-1], symbol)
                    else:
                        try:
                            v = self.COMMANDS[symbol]()
                        except:
                            if symbol == '?':
                                try:
                                    v = inputs[I]
                                    I += 1
                                except:
                                    raise NoMoreInputError(line, code[line-1])
                            else:
                                v = None
                        if v is None:
                            continue
                        self.x = v

        if not self.called and self.functions:
            func = self.functions[list(self.functions.keys())[0]]
            if I < len(inputs):
                result = func(*inputs[I:])
            else:
                result = func()
            if type(result) != Null and not self.implicit:
                print(result)

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
                'D':self}

    def add(self,y):
        return self.x + y

    def minus(self,y):
        return self.x - y

    def times(self,y):
        return self.x * y

    def divide(self,y):
        return self.x / y
    
    def int_divide(self,y):
        return int(self.x / y)

    def power(self,y):
        return self.x ** y

    def greater(self,y):
        return int(self.x > y)

    def less(self,y):
        return int(self.x < y)
    
    def factorial(self):
        return math.factorial(self.x)

    def modulo(self,y):
        return self.x % y

    def negative(self):
        return -self.x

    def equal(self,y):
        return int(self.x == y)

    def next(self):
        self.string += chr(self.x)

    def double(self):
        return self.x * 2

    def half(self):
        return self.x / 2

    def notequals(self,y):
        return int(self.x != y)

    def not_(self):
        return int(not self.x)

    def print(self):
        if self.string:
            print(self.string)
        else:
            print(chr(self.x))

    def print_(self):
        print(self.x)
        
    def _print(self):
        if self.string:
            print(end=self.string)
        else:
            print(end=chr(self.x))

    def randint(self,y=0):
        if y > self.x:
            return random.randint(self.x, y)
        return random.randint(y, self.x)
    
    def sqrt(self):
        return math.sqrt(self.x)
        
    def store(self):
        self.stored.append(self.x)
        
    def get(self):
        self.x = self.stored.pop()
