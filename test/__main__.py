import argparse

### The testing function!
def testall(*args, **kwargs):
    pass

### Entrypoint
def main():
    ### Called as a script, run testall
    args = []
    kwargs = {}

    testall(args, kwargs)

### Invoke the main function directly - we don't need to wrap __main__.py with if __name__ == "__main__"
### See https://docs.python.org/3/library/__main__.html
main()