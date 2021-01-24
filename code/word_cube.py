import numpy as np
import copy
from cube_checker import CubeChecker
from datetime import datetime as dt


class InvalidMoveException(Exception):
    pass


class WordCube:
    """This class provides a representation of a 4x4 cube with letters on it
    """

    # initial positions of letters in the magic cube array
    LID_INITIALISATION = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 35, 39, 43, 47, 34, 38, 42,
                          46, 33, 37, 41, 45, 32, 36, 40, 44, 95, 91, 87, 83, 94, 90, 86, 82, 93, 89, 85, 81,
                          92, 88, 84, 80, 63, 59, 55, 51, 62, 58, 54, 50, 61, 57, 53, 49, 60, 56, 52, 48, 75,
                          74, 73, 79, 72, 71, 70, 78, 69, 68, 67, 77, 66, 65, 64, 76, 28, 29, 30, 31, 24, 25,
                          26, 27, 20, 21, 22, 23, 16, 17, 18, 19]

    LANGUAGES = ["de", "en", "es", "eu"]
    FACES = ["U", "L", "F", "R", "B", "D"]

    # adjacent faces are defined when looking at the face, so that the first of the adjacent faces in the list is
    # at the '12-o'clock' position
    ADJ_FACES = {
        "U": ("B", "R", "F", "L"),
        "L": ("U", "F", "D", "B"),
        "R": ("U", "B", "D", "F"),
        "F": ("U", "R", "D", "L"),
        "B": ("U", "L", "D", "R"),
        "D": ("F", "R", "B", "L")
    }

    def __init__(self, config=None, alphabet=None, language="es", shuffle=True, seed=42):
        assert(language in self.LANGUAGES)
        self.language = language

        alphabet_es = {
                "E": (12, 1),
                "A": (11, 1),
                "I": (6, 1),
                "O": (9, 1),
                "N": (5, 1),
                "R": (5, 1),
                "T": (4, 1),
                "L": (4, 1),
                "S": (6, 1),
                "U": (5, 1),
                "D": (5, 2),
                "G": (2, 2),
                "B": (2, 3),
                "C": (4, 3),
                "M": (2, 3),
                "P": (2, 3),
                "F": (1, 4),
                "H": (2, 4),
                "V": (1, 4),
                "Y": (1, 4),
                "CH": (1, 5),
                "J": (1, 8),
                "LL": (1, 8),
                "Ñ": (1, 8),
                "RR": (1, 8),
                "X": (1, 8),
                "Z": (1, 10)
            }

        alphabet_en = {
                "E": (12, 1),
                "A": (9, 1),
                "I": (9, 1),
                "O": (8, 1),
                "N": (6, 1),
                "R": (6, 1),
                "T": (6, 1),
                "L": (4, 1),
                "S": (4, 1),
                "U": (4, 1),
                "D": (4, 2),
                "G": (3, 2),
                "B": (2, 3),
                "C": (2, 3),
                "M": (2, 3),
                "P": (2, 3),
                "F": (2, 4),
                "H": (2, 4),
                "V": (2, 4),
                "W": (2, 4),
                "Y": (2, 4),
                "K": (1, 5),
                "J": (1, 8),
                "Z": (1, 10)
        }

        self.alphabet_lookup = {
            "es": alphabet_es,
            "en": alphabet_en
        }

        # alphabet - symbol: (num, cost) alphabet must be language dependent
        # english scrabble (without blanks and without Q and X)
        if alphabet is None:
            self.alphabet = self.alphabet_lookup[language]

        if config is None:
            self.config = np.asarray([Letter(k) for k, v in self.alphabet.items() for i in range(v[0])])
        else:
            self.config = config

        if shuffle:
            randomizer = np.random.default_rng(seed=seed)
            randomizer.shuffle(self.config)

        # add lids to the letters for finding the labels in the cube_interactive
        for i, letter in enumerate(self.config):
            letter.set_lid(self.LID_INITIALISATION[i])

        # cube is represented by 6 faces with letters in a 4x4 square
        self.config = self.config.reshape((6, 4, 4))

        # add ability to check for words on the cube depending on the language

        self.checker = CubeChecker(self.language, self.alphabet) # att fst object?

    def make_move(self, face, layer, direction="cw"):
        try:
            assert (0 <= layer <= 3)
            assert (face in self.FACES)
            assert (direction == "cw" or direction == "ccw")
        except AssertionError:
            raise InvalidMoveException()

        face_idx = self.FACES.index(face)
        back_face = list(set(self.FACES) - set(self.ADJ_FACES[face]) - set(face))[0]

        # translate move into same move from the opposite side, if one of the 3 sides is chosen
        # this makes it necessary only to implement half of all possible moves, as you can mirror any move to the
        # opposite side
        if face in ["R", "B", "D"]:
            # print("original move:", face, layer, direction)
            self.make_move(back_face, 3 - layer, "ccw" if (direction == "cw") else "cw")
            return

        # rotate the  opposite face in the opposite direction
        if layer == 3:
            # get the face on the other side
            back_face_idx = self.FACES.index(back_face)
            # define rotation in the opposite direction
            # 1 is counter-clock-wise, 3 is clockwise
            rot_times = 1 if direction == "cw" else 3
            self.config[back_face_idx] = np.rot90(self.config[back_face_idx], rot_times)

        if layer == 0:
            # in case the outer most layer is rotated we have to manipulate 5 faces
            # get the face which needs to be rotated
            rot_times = 3 if direction == "cw" else 1
            self.config[face_idx] = np.rot90(self.config[face_idx], rot_times)

        # print("Move:", face, layer, direction)

        # exchange adjacent faces
        if face == "U":
            # top row of B
            if direction == "cw":
                tmp = copy.deepcopy(self.config[4, layer, :])
                self.config[4, layer, :] = self.config[1, layer, :]
                self.config[1, layer, :] = self.config[2, layer, :]
                self.config[2, layer, :] = self.config[3, layer, :]
                self.config[3, layer, :] = tmp
            else:
                tmp = copy.deepcopy(self.config[4, layer, :])
                self.config[4, layer, :] = self.config[3, layer, :]
                self.config[3, layer, :] = self.config[2, layer, :]
                self.config[2, layer, :] = self.config[1, layer, :]
                self.config[1, layer, :] = tmp
        elif face == "L":
            if direction == "cw":
                tmp = copy.deepcopy(self.config[0, :, layer])
                self.config[0, :, layer] = np.flip(self.config[4, :, 3 - layer])
                self.config[4, :, 3 - layer] = np.flip(self.config[5, :, layer])
                self.config[5, :, layer] = self.config[2, :, layer]
                self.config[2, :, layer] = tmp
            else:
                tmp = copy.deepcopy(self.config[0, :, layer])
                self.config[0, :, layer] = self.config[2, :, layer]
                self.config[2, :, layer] = self.config[5, :, layer]
                self.config[5, :, layer] = np.flip(self.config[4, :, 3 - layer])
                self.config[4, :, 3 - layer] = np.flip(tmp)
        elif face == "F":
            if direction == "cw":
                tmp = copy.deepcopy(self.config[0, 3 - layer, :])
                self.config[0, 3 - layer, :] = np.flip(self.config[1, :, 3 - layer])
                self.config[1, :, 3 - layer] = self.config[5, layer, :]
                self.config[5, layer, :] = np.flip(self.config[3, :, layer])
                self.config[3, :, layer] = tmp
            else:
                tmp = copy.deepcopy(self.config[0, 3 - layer, :])
                self.config[0, 3 - layer, :] = self.config[3, :, layer]
                self.config[3, :, layer] = np.flip(self.config[5, layer, :])
                self.config[5, layer, :] = self.config[1, :, 3 - layer]
                self.config[1, :, 3 - layer] = np.flip(tmp)

    def get_strings(self, check_face):
        horizontal = None
        vertical = None

        if check_face in ["L", "F", "R", "B"]:
            horizontal = np.array([np.concatenate(self.config[1:5, i, :]) for i in range(self.config.shape[1])])
        if check_face == "U":
            pass
        if check_face == "D":
            pass

        if check_face in ["U", "F", "D", "B"]:
            vertical = np.array(
                [np.concatenate([np.concatenate(self.config[[0, 2, 5], :, i]), self.config[4, ::-1, 3-i]]) for i in
                 range(self.config.shape[2])])
            if check_face == "B":
                vertical = np.array([np.flip(arr) for arr in vertical[::-1]])
        if check_face == "L":
            pass
        if check_face == "R":
            pass

        return np.concatenate((vertical, horizontal))

    def check_cube(self):
        lines = self.get_strings("F")

        for line in lines:
            print("".join([l.get_s() for l in line]))

        strings = ["".join([letter.get_s() for letter in line]) for line in lines]
        ids = [[letter.get_lid() for letter in line] for line in lines]

        # print(strings)
        # print(ids)

        found_strings = list()
        for i, s in enumerate(strings):
            found_strings.append((self.checker.check_string(s), i))

        # print(found_strings)
        # print(found_strings)

        top_match = ((-1,""),-1)
        for matches in found_strings:
            i = matches[1]
            if len(matches) == 0:
                continue
            for match in matches[0]:
                # print(match)
                if top_match is not None and top_match[0][0] >= match[0]:
                        continue
                # make sure the word fits on the cube
                elif len(match[1]) <= 16 and len(match[1]) > 3:
                    top_match = (match, i)
        # find identified sequence in the cube
        # make sure the match is no longer than 16 characters
        # make sure to know where the match starts and ends
        print("top word", top_match[0][1])
        start_idx = (strings[top_match[1]]).find(top_match[0][1].upper())
        # calculate the actual length for words on the cube in terms of letters (some letters have multiple characters)
        letters = lines[top_match[1]][start_idx:start_idx+len(top_match[0][1])]
        letters_len = sum([len(l.get_s()) for l in letters])
        word_len_on_cube = len(letters) - np.abs(letters_len - len(top_match[0][1]))

        print(word_len_on_cube)

        cube_idxs = ids[top_match[1]][start_idx:start_idx+word_len_on_cube]

        return cube_idxs

    def __str__(self):
        spacing = "\t\t\t"

        ret = spacing

        for r in self.config[0]:
            for c in r:
                ret += "|{}".format(c)
            ret += "|\n" + spacing
        ret += "\n"

        for i in range(self.config.shape[1]):
            for j, r in enumerate(self.config[1:-1, i, :]):
                for c in r:
                    ret += "|{}".format(c)
                if j % 4 != 3:
                    ret += "|\t"
            ret += "|\n"
        ret += "\n" + spacing

        for r in self.config[-1]:
            for c in r:
                ret += "|{}".format(c)
            ret += "|\n" + spacing

        return ret


class Letter:
    def __init__(self, s, lid=None):
        self.s = s
        self.lid = lid

    def lower(self):
        return self.s.lower()

    def upper(self):
        return self.s.upper()

    def get_lid(self):
        return self.lid

    def get_s(self):
        return self.s

    def set_lid(self, lid):
        self.lid = lid

    def set_s(self, s):
        self.s = s

    def __str__(self):
        return self.s# + str(self.lid)

    def __eq__(self, other):
        try:
            assert type(other) == Letter
            assert other.s == self.s
            return True
        except AssertionError:
            return False


if __name__ == "__main__":
    wc = WordCube(shuffle=False)

    print(wc)
    """
    config = np.asarray([[['P', 'E', 'L', 'E'],
        ['T', 'R', 'A', 'B'],
        ['I', 'I', 'Z', 'N'],
        ['G', 'B', 'W', 'O']],
        [['P', 'S', 'S', 'S'],
        ['T', 'R', 'I', 'I'],
        ['C', 'S', 'O', 'A'],
        ['Y', 'A', 'E', 'M']],
        [['P', 'E', 'L', 'E'],
        ['I', 'W', 'D', 'R'],
        ['I', 'A', 'E', 'S'],
        ['L', 'U', 'K', 'N']],
        [['N', 'T', 'I', 'S'],
        ['O', 'C', 'E', 'E'],
        ['H', 'L', 'E', 'F'],
        ['N', 'T', 'V', 'P']],
        [['A', 'E', 'F', 'O'],
        ['O', 'T', 'V', 'H'],
        ['L', 'G', 'O', 'I'],
        ['U', 'O', 'N', 'S']],
        [['J', 'N', 'D', 'E'],
        ['N', 'D', 'E', 'D'],
        ['R', 'E', 'E', 'U'],
        ['I', 'R', 'M', 'O']]], dtype=object)

    wc = WordCube(config=config, shuffle=False)

    print(wc)

    h, v = wc.get_strings(check_face="F")

    start = dt.now()
    cc = cube_checker.CubeChecker("att_files/zapChecker.att", wc.alphabet)


    res = list()
    for word in h:
        word = "".join(word)
        word = word[1:] + word + word[:-1]
        res.append(cc.check_string(word))
    for word in v:
        word = "".join(word)
        word = word[1:] + word + word[:-1]
        res.append(cc.check_string(word))

    print(res)

    print(dt.now()-start)
    """
