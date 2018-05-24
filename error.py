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

class IncongruentTypesError(Exception):
    def __init__(self, num, line, command):

        self.message = '''Fatal error: IncongruentTypesError
    line {}: '{}'
        Unable to perform the command '{}' due to inconsistent type operands'''.format(num, line, command)

        super(IncongruentTypesError, self).__init__(self.message)

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
    def __init__(self, num, line, char = None):
        
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

class InvalidQuoteSyntaxError(Exception):
    def __init__(self, num, line):

        self.message = '''Fatal error: InvalidQuoteSyntaxError
    line {}: '{}'
        Quotes must be escaped in strings'''.format(num, line)

        super(InvalidQuoteSyntaxError, self).__init__(self.message)

class InvalidArgumentError(Exception):
    def __init__(self, num, line, arg = None):

        self.message = '''Fatal error: InvalidArgumentError
    line {}: '{}'
        {} is an invalid function argument'''.format(num, line, arg)

        super(InvalidArgumentError, self).__init__(self.message)

class InvalidSyntaxError(Exception):
    def __init__(self, num, line):

        self.message = '''Fatal error: InvalidSyntaxError
    line {}: '{}'
        Invalid syntax'''.format(num, line)

        super(InvalidSyntaxError, self).__init__(self.message)

class UnknownVariableError(Exception):
    def __init__(self, num, line, var):

        self.message = '''Fatal error: UnknownVariableError
    line {}: '{}'
        Unknown variable reference: '{}'.
        Funcargs should be defined with @#<var> to keep'''.format(num, line, var)

        super(UnknownVariableError, self).__init__(self.message)

class PythonError(Exception):
    def __init__(self, num, line, error):

        self.message = '''Fatal error: PythonError
    line {}: '{}'
        Python error raised: {}'''.format(num, line, error)

        super(PythonError, self).__init__(self.message)
