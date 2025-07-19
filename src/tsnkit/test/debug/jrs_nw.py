from .. import debug

if __name__ == "__main__":
    args = debug.parse()
    debug.run("jrs_nw", args)
