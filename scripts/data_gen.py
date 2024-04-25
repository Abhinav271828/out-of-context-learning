from tqdm import tqdm

def gen_dyck_n_strings(l, open, close):
    """
    Generate valid bracket sequences of length 2l
    using open bracket `open` and close bracket `close`.
    """
    if l == 0:
        return [""]
    if l == 1:
        return [open + close]
    res = []
    for i in range(l):
        for x in gen_dyck_n_strings(i, open, close):
            for y in gen_dyck_n_strings(l - i - 1, open, close):
                res.append(open + x + close + y)
    return res


def shuffle(s1, s2):
    """
    Return all possible ways to shuffle two strings s1 and s2
    such that the order of characters in each string is preserved.
    """
    l = len(s1) + len(s2)

    shuffle = []
    for i in range(2**l):
        bits = bin(i)[2:]
        if len(bits) < l:
            bits = "0" * (l - len(bits)) + bits
        if bits.count("0") != len(s1) or bits.count("1") != len(s2):
            continue

        l1 = list(s1)
        l2 = list(s2)
        string = "".join(l1.pop(0) if bit == "0" else l2.pop(0) for bit in bits)
        shuffle.append(string)
    return shuffle


def generate_shuffle_dyck_2(length):
    """
    Generate all possible shuffles of two Dyck-n strings (using different
    brackets) with total length `length`.
    Write these strings to a file `shuffle_dyck_2.txt`.
    """
    with open("shuffle_dyck_2.txt", "w") as f:
        for l1 in range(length + 1):
            l2 = length - l1
            for s1 in gen_dyck_n_strings(l1, "(", ")"):
                for s2 in tqdm(gen_dyck_n_strings(l2, "[", "]"), desc=f"Lengths {l1}+{l2}"):
                    for string in shuffle(s1, s2):
                        f.write(string + "\n")

generate_shuffle_dyck_2(8)