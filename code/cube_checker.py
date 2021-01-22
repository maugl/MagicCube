import attapply
import word_cube


class CubeChecker:
    def __init__(self, checkerFile, alphabet):
        self.transducer = attapply.ATTFST(checkerFile)
        self.alphabet = {k:v[1] for k,v in alphabet.items()}

    def check_string(self, input_string):
        check_string = input_string[1:] + input_string + input_string[:-1]
        words = list(self.transducer.apply(word=check_string.lower(), dir="down"))
        return self.weight_words(words)

    def weight_words(self, words):
        result = list()
        for w, _ in words:
            sum = 0
            for l in w:
                sum += self.alphabet[l.upper()]
            result.append((sum, w))
        return list((sorted(set(result))))


if __name__ == "__main__":
    wc = word_cube.WordCube()
    cc = CubeChecker("att_files/zapChecker.att", wc.alphabet)

    print(cc.check_string("asdklblaspelensblasdasd"))
