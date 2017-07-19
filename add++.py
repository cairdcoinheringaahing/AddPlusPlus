import math, random

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
    def push(self,*values):
        for v in values:
            self.append(v)
    def pop(self,index=-1):
        try:
            return super().pop(index)
        except:
            return -1
        
def add(x,y):
    if type(x) == str:
        return x + str(y)
    if type(y) == int:
        return x + y
    return x

def subtract(x,y):
    if type(x) == str:
        x = list(x)
        for c in str(y):
            x.remove(c)
        return "".join(x)
    if type(y) == int:
        return x - y
    return x

def multiply(x,y):
    if type(x) == str and type(y) == str:
        final = ""
        for a,b in zip(x,y):
            final += chr(ord(a)+ord(b))
        return final
    return x * y

def divide(x,y):
    if type(x) == str:
        if type(y) == str:
            final = ""
            for a,b in zip(x,y):
                final += chr(max(ord(a),ord(b))-min(ord(a),ord(b)))
            return final
        return x[:len(x)//y]

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
        return [y,x][len(x)>len(y)][:min(len(x),len(y))]
    if type(x) == str and type(y) == int:
        return x[::y][-1] if len(x)%2 == 1 else ""
    if type(x) == int and type(y) == str:
        return y[::x][-1] if len(y)%2 == 1 else ""
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
    
class StackScript:

    def __init__(self,code,args,stack=Stack()):
        self.args = args
        self.stack = stack
        self.code = StackScript.tokenize(code)
        cont = False
        for cmd in self.code:
            if cont:
                continue
            if cmd[0] == '"':
                self.stack.push(cmd[1:])
            elif isdigit(cmd):
                self.stack.push(eval_(cmd))
            else:
                if cmd == 'r':
                    if self.stack.pop():
                        cont = True
                    continue
                command = self.COMMANDS[cmd]
                command()

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
        return {'+':lambda: self.stack.push(add(self.stack.pop(), self.stack.pop())),
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
                ':':lambda: self.stack.push(self.stack[-2]),
                'A':lambda: self.stack.push(*self.args),
                'N':lambda: self.stack.push('\n'.join(map(str, self.stack)))
               }
    
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
        stack = list(filter(lambda x: x%2 == int(bool(even_odd)), self.stack))
        self.stack.clear()
        self.stack.push(*stack)

class Function:

    def __init__(self,name,args,code,return_flag,text):
        self.name = name
        self.args = args
        self.code = code
        self.stack = Stack()
        self.flag = return_flag
        self.text = text

    def __call__(self,*args):
        args = list(args)
        while len(args) != self.args:
            args.append(-1)
        self.stack.push(*args)
        script = StackScript(self.code,args,self.stack)
        value = script.run(self.flag,self.text)
        return value
        
    def __repr__(self):
        return 'function {} that takes {} arguments and contains the code {}'.format(self.name,self.args,self.code)

class Script:

    def __init__(self,code,inputs=(),x=0,recur=False):
        
        code = list(map(lambda x: x.split(','), filter(None, code.split('\n'))))
            
        self.string = ''
        self.functions = {}
        if not recur:
            self.stored = []
        I = 0

        self.x = x
        self.y = 0
        
        f = code[:]
        code.clear()
        for i in range(len(f)):
            if len(f[i]) > 1:
                code.append(f[i])
            else:
                code.append(f[i][0])

        if code.count(':') > 1:
            return None
        
        if ':' in code[0]:
            i = 1
            assign = code[0].split(':')[1]
            if assign == '?':
                try:
                    self.x = inputs[I]
                except:
                    self.x = 0
                I += 1
            else:
                self.x = eval_(assign)
        else:
            i = 0
            try:
                if code[0] not in 'FIEWD':
                    self.x = x
                    self.code = code
            except:
                if code[0][0] not in 'FEIWD':
                    self.x = x
                    self.code = code
            
        for cmd in code[i:]:
            if type(cmd) == list:
                if cmd[0] == 'F':
                    for i in range(int(self.x)):
                        self.__init__('\n'.join(cmd),inputs,recur=True)
                if cmd[0] == 'E':
                    for i in range(int(self.x)):
                        self.__init__('\n'.join(cmd),inputs,i,recur=True)
                if cmd[0] == 'I':
                    if self.x:
                        self.__init__('\n'.join(cmd),inputs,recur=True)
                if cmd[0] == 'W':
                    while self.x:
                        self.__init__('\n'.join(cmd),inputs,self.x,recur=True)
                if cmd[0] == 'D':
                    func_name = cmd[1]
                    if func_name in 'NPORSFIWD':
                        return None
                    func_args = cmd[2].count('@')
                    return_flag = '*' in cmd[2]
                    text_flag = '^' in cmd[2]
                    func_code = ''.join(cmd[3:])
                    self.functions[func_name] = Function(func_name,func_args,func_code,return_flag,text_flag)
                if cmd[0][0] == '$':
                    func = self.functions[cmd[0][1:]]
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
                        else:
                            args.append(eval_(c))
                    value = func(*args)
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

                if value:
                    if value == '?':
                        value = inputs[I]
                        I += 1
                    if value == 'G':
                        try:
                            value = self.stored.pop()
                        except:
                            pass
                    if value == 'x':
                        value = self.x
                    if value == 'y':
                        value = self.y
                    try:
                        self.x = self.COMMANDS[symbol](value)
                    except:
                        continue
                else:
                    try:
                        v = self.COMMANDS[symbol]()
                    except:
                        if symbol == '?':
                            v = inputs[I]
                            I += 1
                        else:
                            v = None
                    if v is None:
                        continue
                    self.x = v

    def __call__(self,*values):
        return None

    @property
    def COMMANDS(self):
        return {'+':self.add,
                '-':self.minus,
                '*':self.times,
                '/':self.divide,
                '\\':self.int_divide,
                '^':self.power,
                '>':self.double,
                '<':self.half,
                '!':self.factorial,
                '%':self.modulo,
                '~':self.negative,
                '=':self.equal,
                '&':self.next,
                
                'N':self.not_,
                'P':self.print,
                'O':self.print_,
                'H':self._print,
                'R':self.randint,
                'S':self.sqrt,
                'Q':self.quine,
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
        return self.x // y

    def power(self,y):
        return self.x ** y

    def double(self):
        return self.x * 2

    def half(self):
        return self.x / 2
    
    def factorial(self):
        return math.factorial(self.x)

    def modulo(self,y):
        return self.x % y

    def negative(self):
        return -self.x

    def equal(self,y):
        return self.x == y

    def not_(self):
        return not self.x

    def print(self):
        if self.string:
            print(self.string)
        else:
            print(chr(self.x))
            self.x = 0

    def print_(self):
        print(self.x)
        
    def _print(self):
        if self.string:
            print(end=self.string)
        else:
            print(end=chr(self.x))

    def randint(self,y=0):
        return random.randint(y, self.x)
    
    def sqrt(self):
        return math.sqrt(self.x)
    
    def quine(self):
        print('\n'.join(self.code))
        
    def store(self):
        self.stored.append(self.x)
        
    def get(self):
        self.x = self.stored.pop(-1)

    def next(self):
        self.string += chr(self.x)

if __name__ == '__main__':

    import sys

    program = sys.argv[1]
    inputs = list(map(eval_, sys.argv[2:]))
    try:
        if program.endswith('.txt'):
            Script(open(program).read(),inputs)
        else:
            Script(program,inputs)
    except:
        print('Error encountered!')
