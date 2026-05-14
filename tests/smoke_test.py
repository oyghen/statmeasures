import statmeasures


def main() -> None:
    result = statmeasures.__name__
    expected = "statmeasures"
    if result == expected:
        print(f"Smoke test for {statmeasures.__name__}: PASSED")
    else:
        raise RuntimeError(f"Smoke test for {statmeasures.__name__}: FAILED")


if __name__ == "__main__":
    main()
