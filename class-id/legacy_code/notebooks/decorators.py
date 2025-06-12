'''
This is a helper file consist of useful function decorators:
Credits: https://medium.com/python-in-plain-english/five-python-wrappers-that-can-reduce-your-code-by-half-af775feb1d5
1- Timer Wrapper
2- Debugger Wrapper
3- Exception Handler Wrapper
4- Input Validator Wrapper
5- Function Retry Wrapper
'''

import time

def timer(func):
    '''
    This wrapper function measures the execution time of a function and prints the elapsed time. It can be useful for profiling and optimizing code.
    Example Usage:
    @timer
    def train_model():
        print("Starting the model training function...")
        # simulate a function execution by pausing the program for 5 seconds
        time.sleep(5) 
        print("Model training completed!")
    
    train_model()
    '''
    def wrapper(*args, **kwargs):
        # start the timer
        start_time = time.time()
        # call the decorated function
        result = func(*args, **kwargs)
        # remeasure the time
        end_time = time.time()
        # compute the elapsed time and print it
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:3f} seconds")
        # return the result of the decorated function execution
        return result
    # return reference to the wrapper function
    return wrapper


def debugIO(func):
    '''
    An additional useful wrapper function can be created to facilitate debugging by printing the inputs and outputs of each function. This approach allows us to gain insight into the execution flow of various functions without cluttering our applications with multiple print statements.
    Example Usage:
    @debug
    def add_numbers(x, y):
        return x + y
    add_numbers(7, y=5,)  # Output: Calling add_numbers with args: (7) kwargs: {'y': 5} \n add_numbers returned: 12
    '''
    def wrapper(*args, **kwargs):
        # print the fucntion name and arguments
        print(f"Calling {func.__name__} with args: {args} kwargs: {kwargs}")
        # call the function
        result = func(*args, **kwargs)
        # print the results
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper



def exception_handler(func):
    '''
    The exception_handler the wrapper will catch any exceptions raised within the divide function and handle them accordingly. We can customize the handling of exceptions within the wrapper function as per requirements, such as logging the exception or performing additional error-handling steps.
    Example Usage:
    @exception_handler
    def divide(x, y):
        result = x / y
        return result
    divide(10, 0)  # Output: An exception occurred: division by zero
    '''
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Handle the exception
            print(f"An exception occurred: {str(e)}")
            # Optionally, perform additional error handling or logging
            # Reraise the exception if needed
    return wrapper



def validate_input(*validations):
    '''
    This wrapper function validates the input arguments of a function against specified conditions or data types. It can be used to ensure the correctness and consistency of the input data. It is important to ensure that the order of the validation functions corresponds to the order of the arguments they 
    are intended to validate.
    Example Usage:
    @validate_input(lambda x: x > 0, lambda y: isinstance(y, str))
    def divide_and_print(x, message):
        print(message)
        return 1 / x
    
    divide_and_print(5, "Hello!")  # Output: Hello! 1.0
    '''
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i, val in enumerate(args):
                if i < len(validations):
                    if not validations[i](val):
                        raise ValueError(f"Invalid argument: {val}")
            for key, val in kwargs.items():
                if key in validations[len(args):]:
                    if not validations[len(args):][key](val):
                        raise ValueError(f"Invalid argument: {key}={val}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry(max_attempts, delay=1):
    '''
    This wrapper retries the execution of a function a specified number of times with a delay between retries. It can be useful when dealing with network or API calls that may occasionally fail due to temporary issues.
    Example Usage:
    @retry(max_attempts=3, delay=2)
    def fetch_data(url):
        print("Fetching the data..")
        # raise timeout error to simulate a server not responding..
        raise TimeoutError("Server is not responding.")
    fetch_data("https://example.com/data")  # Retries 3 times with a 2-second delay between attempts
    '''
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed: {e}")
                    time.sleep(delay)
            print(f"Function failed after {max_attempts} attempts")
        return wrapper
    return decorator