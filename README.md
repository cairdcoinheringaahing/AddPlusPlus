# Add++

A programming language with advanced features, such as functions.

You can install Add++ through the generic way of downloading code from GitHub (click "Clone or download" up to the right).

Or if you don't want to download it, you can run it [here](https://tio.run/#addpp) thanks to Dennis Mitchell.

For a full list of commands, please read the [wiki](https://github.com/cairdcoinheringaahing/AddPlusPlus/wiki)

If you wish to run Add++ locally, the recommended invocation is

    python testsuite.py
    python add++.py <flags> <file> <argv>

This will check to make sure that no errors are found with the lastest version. If any output besides `No failures` comes from the first call, please [file an issue](https://github.com/cairdcoinheringaahing/AddPlusPlus/issues/new), including the full output text.

You can also use the call

    python add++.py -h
    
To get help on the command line invocation of the Add++ file.

If you have both Python 2 and 3 installed on your computer, change the invocations to

    python3 testsuite.py
    python3 add++.py <flags> <file> <argv>
