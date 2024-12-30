# for each line in build/used_symbols.txt
# find defined_symbols


def main():
    needed_symbols = []
    o_files_to_use = []
    final_o_files = []
    not_found_symbols = []
    with open("build/used_symbols.txt", "r") as f:
        for function_name in f:
            needed_symbols.append(function_name.strip())
    declared_symbols_to_file = {}
    file_to_undeclared_symbols = {}
    with open("build/defined_libtorch_cpu.txt", "r") as f:
        o_file = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                o_file = line[:-1]
                continue
            if o_file is None:
                continue
            declared_symbol = line.split()[-1]
            declared_symbols_to_file[declared_symbol] = o_file
    # load dependencies of each o file
    with open("build/undefined_libtorch_cpu.txt", "r") as f:
        o_file = None
        undeclared_symbol_buffer = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                if o_file is not None:
                    file_to_undeclared_symbols[o_file] = undeclared_symbol_buffer
                    undeclared_symbol_buffer = []
                o_file = line[:-1]
                continue
            if o_file is None:
                continue
            undeclared_symbol = line.split()[-1]
            undeclared_symbol_buffer.append(undeclared_symbol)

    for needed_symbol in needed_symbols:
        if needed_symbol in declared_symbols_to_file:
            print(f"{needed_symbol} found in {declared_symbols_to_file[needed_symbol]}")
            o_files_to_use.append(declared_symbols_to_file[needed_symbol])
        else:
            print("not found:", needed_symbol)
            not_found_symbols.append(needed_symbol)
    needed_symbols = set()
    final_o_files.extend(o_files_to_use)

    while True:
        # for each o_file, find the dependencies
        for o_file in o_files_to_use:
            if o_file in file_to_undeclared_symbols:
                print(f"Dependencies of {o_file}:")
                for symbol in file_to_undeclared_symbols[o_file]:
                    needed_symbols.add(symbol)
            else:
                print(f"No dependencies found for {o_file}")

        final_o_files.extend(o_files_to_use)
        o_files_to_use = []

        for needed_symbol in needed_symbols:
            if needed_symbol in declared_symbols_to_file:
                # print(f"{needed_symbol} found in {declared_symbols_to_file[needed_symbol]}")
                if declared_symbols_to_file[needed_symbol] not in final_o_files:
                    o_files_to_use.append(declared_symbols_to_file[needed_symbol])
            else:
                not_found_symbols.append(needed_symbol)
        needed_symbols = set()

        if not o_files_to_use:
            break

    print("Final o files:")
    final_o_files = list(set(final_o_files))
    print(final_o_files)
    print("Total: ", len(final_o_files))
    with open("needed_objects.txt", "w") as f:
        for o_file in final_o_files:
            f.write(o_file + "\n")
    print("Not found symbols:")
    not_found_symbols = list(set(not_found_symbols))
    print(not_found_symbols)
    print("Total: ", len(not_found_symbols))
    with open("not_found_symbols.txt", "w") as f:
        for symbol in not_found_symbols:
            f.write(symbol + "\n")


if __name__ == "__main__":
    main()
