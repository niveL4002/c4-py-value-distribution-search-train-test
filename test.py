import mysearch
import random

def main():
    for n_slots in [3, 5, 9, 13]:
        print("n slots:", n_slots)
        # get children
        children = random_children(10, n_slots)

        print("children:")
        for i, child in enumerate(children):
            print(i + 1, ":", mysearch.get_D_str(child.D))

        # estimate
        weights = mysearch.get_selection_weights(children)
        normalize(weights)

        # thompson distribution
        thompson = [0]*len(children)
        for _ in range(1000_000):
            thompson[mysearch.thompson_select(children)] += 1
        normalize(thompson)

        # result
        print("weights :", [round(x, 4) for x in weights])
        print("thompson:", [round(x, 4) for x in thompson])
        print("abs_error:", abs_error(weights, thompson),"\n")

def abs_error(A, B) -> float:
    return sum(abs(a - b) for a, b in zip(A, B))

def normalize(A):
    s = sum(A)
    for i in range(len(A)):
        A[i] /= s


def random_children(n, D_len) -> [mysearch.Node]:
    children = []
    for _ in range(n):
        D = [random.random() for _ in range(D_len)]
        normalize(D)
        children.append(mysearch.Node(D=D))
    return children

if __name__ == "__main__":
    main()
