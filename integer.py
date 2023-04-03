if __name__ == "__main__":
    mm = -2147483648
    mp = 2147483647
    total = 0
    data = [[mm, mm], [mp, mm], [mp, mp], [mm, mp]]

    def cross(a, b):
        return a[0] * (b[1] - a[1]) - a[0] * (b[0] - a[0])

    i = 0
    for i in range(len(data)):
        print(i)
        j = (i + 1) % len(data)
        p1 = data[i]
        p2 = data[j]
        c = cross(p1, p2)
        print(f"p1:{p1}, p2:{p2}, c:{c}")
        total += c
        print(f"total:{total}")
