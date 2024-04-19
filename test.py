global_variable = 10

def change_global():
    global global_variable
    global_variable += 1

def main():
    global global_variable
    print("Before the loop:", global_variable)
    while global_variable < 15:
        change_global()
    print("After the loop:", global_variable)

if __name__ == "__main__":
    main()