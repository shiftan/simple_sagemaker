import time


def main():
    vals1 = [60, 6, 0.6]
    vals2 = [-10, 0, 10]
    for (val1, val2) in zip(vals1, vals2):
        print(f"Val1: {val1:.4e}")
        print(f"Val2: {val2}")
        time.sleep(60)


if __name__ == "__main__":
    main()
