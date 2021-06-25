if __name__ == '__main__':
    import argparse as ap

    parser = ap.ArgumentParser()
    parser.add_argument("-F", "--files", nargs="+", required=True)

    args = parser.parse_args()
    print(args.files)
