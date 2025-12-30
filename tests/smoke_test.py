import statmeasures


def main():
    result = statmeasures.__name__
    expected = "statmeasures"
    if result == expected:
        print("smoke test passed")
    else:
        raise RuntimeError("smoke test failed")


if __name__ == "__main__":
    main()
