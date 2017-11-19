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
