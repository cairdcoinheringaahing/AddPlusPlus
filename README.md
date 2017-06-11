# Add++ [frozen-development]

**Add++ is no longer being updated and will stay as it is. No more features will be added and any existing bugs will remain**

Add++ is a functional, line by line programming language that may or may not be Turing complete.

The most notable thing about Add++ is its *functions*. Let's go over them now.

## Functions

Functions follow a certain syntax to be considered valid code

    D,<name of function>,<arguments>,<code>
    
### Name of the function

You can name the function anything you want. Literally anything, except for the single letters `FIDEW` as these are special commands

### Arguments

The number of arguments for a function are defined as the number of `@` in the `<arguments>` section. Each `@` represents 1 argument. If you include a `*` in the arguments the entire stack is returned from the function, rather than just the top item. An `^` being included means that it returns the stack as a string

**TAKE NOTE! These two symbol commands are currently under development; dp not use with certainty**

### Code

Functions use a stack memory system, compared to `main` which uses an accumulator (`x`)

As such, many commands from `main` are also valid commands in functions.

As the stack contains both strings and numbers, commands can be adapted to work with either one. For example, `+` will add two numbers together but will concatinate two strings or convert an into into a string and concatinate that with another string.

Commands that do this are

    commands int/string

    +  Addition/concatination
    -  Subtraction/removing characters
    *  Multiplying numbers/multiplying char points
    /  Divide/split into substrings
    %  Modulo/Get left-over characters from string
    ^  Exponention/elementwise maximum
    
An example to concatinate two strings/add teo numbers, depending on arguments could be

    D,add or concatinate,@@,+
    
## Calling functions

Functions are called using `$` syntax with arguments following as comma separted values e.g

    $add or concatinate,10,20
    
would call the above function with the arguments of `10` and `20`.

`x` is automatically set to the result of the function application.

*More notes to come; this is just the beginning*
