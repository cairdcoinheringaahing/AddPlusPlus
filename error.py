import sys

def base(name, num, line, reason):
    message = '''Fatal error: {}Error
    line {}: '{}'
        {}'''.format(name, num, line, reason)

    print(message, file = sys.stderr)
    sys.exit(1)

def DivisionByZeroError(num, line):
    base('DivisionByZero', num, line, 'Division by 0 is undefined')

def EmptySecondStackError(num, line):
    base('EmptySecondStack', num, line, 'The second stack is empty and is unable to be popped from')

def EmptyStackError(num, line):
    'The stack is empty, and is unable to be popped from'

def IncongruentTypesError(num, line, command):
    reason = "Unable to perform the command '{}' due to inconsistent type operands".format(command)
    base('IncongruentTypes', num, line, reason)

def InvalidArgumentError(num, line, arg = None):
    base('InvalidArgument', num, line, "'{}' is an invalid function argument".format(arg))

def InvalidQuoteSyntaxError(num, line):
    base('InvalidQuoteSyntax', num, line, 'Quotes must be escaped in strings')

def InvalidSymbolError(num, line, char = None):
    base('InvalidSymbol', num, line, "The character '{}' is an invalid Add++ character".format(char))

def InvalidSyntaxError(num, line):
    base('InvalidSyntax', num, line, 'InvalidSyntax')

def NoMoreInputError(num, line):
    base('NoMoreInput', num, line, 'All input has been used and cannot be used again')

def UnableToRetrieveFunctionError(num, line, name):
    reason = "Unable to retrieve function '{}'. Functions must be defined before being called".format(name)
    base('UnableToRetrieveFunction', num, line, reason)

def UnknownVariableError(num, line, var = None):
    reason  = "Unknown variable reference: '{}'.\n".format(var)
    reason += "Funcargs should be defined with @#var to maintain access"

    base('UnknownVariable', num, line, reason)

# =-= #

def PythonError(num, line, error = None):
    reason = 'Python error raised: {}'.format(error)
    base('Python', num, line, reason)
