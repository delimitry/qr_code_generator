#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
QR Code generator

The word "QR Code" is registered trademark of DENSO WAVE INCORPORATED
http://www.denso-wave.com/qrcode/faqpatent-e.html
"""

import itertools
from PIL import Image, ImageDraw
from qr_data_tables import (
    alignment_pattern_locations, error_correction_levels, modes, term_sizes,
    alphanumeric_table, capacity, error_correction, generator_polynomials
)


def alphanumeric_to_binary(data):
    """ Convert alphanumeric to binary """
    if not data:
        return ''
    for char in data:
        if char not in alphanumeric_table.keys():
            raise Exception(
                'Alphanumeric mode support only alphanumeric characters "%s"!' %
                ''.join(sorted(alphanumeric_table.keys())))
    result = ''
    for i in range(0, (len(data) // 2) * 2, 2):
        if len(data[i:i + 2]) > 1:
            result += bin(45 * alphanumeric_table[data[i]] + alphanumeric_table[data[i + 1]])[2:].zfill(11)
        else:
            result += bin(45 * alphanumeric_table[data[i]])[2:].zfill(11)
    if len(data) % 2:
        result += bin(alphanumeric_table[data[-1]])[2:].zfill(6)
    return result


def numeric_to_binary(data):
    """ Convert numeric to binary """
    if not data:
        return ''
    if not data.isdigit():
        raise Exception('Numeric mode support only 0..9 characters!')
    result = ''
    for i in range(0, len(data), 3):
        bin_val = bin(int(data[i:i + 3], 10))[2:]
        if len(data[i:i + 3]) == 3:
            result += bin_val.zfill(10)
        elif len(data[i:i + 3]) == 2:
            result += bin_val.zfill(7)
        else:
            result += bin_val.zfill(4)
    return result


def get_character_count_bits(version, mode):
    """ Get number of bits in character count indicator """
    if 1 <= version <= 9:
        if mode == 'numeric':
            return 10
        elif mode == 'alphanumeric':
            return 9
        elif mode == 'byte':
            return 8
        elif mode == 'kanji':
            return 8
    elif 10 <= version <= 26:
        if mode == 'numeric':
            return 12
        elif mode == 'alphanumeric':
            return 11
        elif mode == 'byte':
            return 16
        elif mode == 'kanji':
            return 10
    elif 27 <= version <= 40:
        if mode == 'numeric':
            return 14
        elif mode == 'alphanumeric':
            return 13
        elif mode == 'byte':
            return 16
        elif mode == 'kanji':
            return 12
    raise Exception('Invalid version and mode! Must be [1..40]')


def get_gf256_table():
    """ Get GF(256) table """
    # fill GF(256) table
    m = 285  # 0b100011101
    exp_a_table = {i: 2 ** i for i in range(8)}
    exp_a_table[8] = 256 ^ m
    for i in range(9, 255):
        val = exp_a_table[i - 1] * 2
        exp_a_table[i] = val ^ m if val >= 256 else val
    return exp_a_table


def get_error_correction_data(data, version, error_correction_level):
    """ Get error correction data """
    exp_a_table = get_gf256_table()
    # get inverted table
    inv_exp_a_table = {}
    for k, v in exp_a_table.items():
        if v not in inv_exp_a_table:
            inv_exp_a_table[v] = k

    error_correction_info = error_correction.get(version, {}).get(error_correction_level, {})
    codewords = error_correction_info.get('codewords')
    if not codewords:
        raise Exception('No codewords!')
    ecc_per_block = error_correction_info.get('ec_codewords_per_block')
    if not ecc_per_block:
        raise Exception('No number error correction codewords!')
    blocks_in_group_1 = error_correction_info.get('blocks_in_group_1')
    blocks_in_group_2 = error_correction_info.get('blocks_in_group_2')
    codewords_in_group_1 = error_correction_info.get('codewords_in_group_1')
    codewords_in_group_2 = error_correction_info.get('codewords_in_group_2')

    def calc_ecc(coeffs):
        """ Calc ECC """
        generator_polynomial = [
            (p, ecc_per_block - j + len(coeffs) - 1) for j, p in enumerate(generator_polynomials.get(ecc_per_block))]
        message_polynomial = [(c, ecc_per_block - j + len(coeffs) - 1) for j, c in enumerate(coeffs)]
        # perform division steps number of terms in message_polynomial times
        for _ in range(len(coeffs) + 1):
            if not (message_polynomial and message_polynomial[0]) or message_polynomial[0][1] < ecc_per_block:
                break
            leading_term = message_polynomial[0][0]
            if leading_term == 0:
                message_polynomial = message_polynomial[1:]
                continue
            # multiply generator_polynomial by leading_term of message_polynomial
            result_gp = [
                (exp_a_table[(inv_exp_a_table[leading_term] + a_pow) % 255], x_pow - len(coeffs) - 1)
                for a_pow, x_pow in generator_polynomial]
            # xor result with message_polynomial
            message_polynomial = [
                (((0 if j >= len(message_polynomial) else message_polynomial[j][0]) ^
                  (0 if j >= len(result_gp) else result_gp[j][0])), message_polynomial[0][1] - j)
                for j in range(max(ecc_per_block + 1, len(message_polynomial)))]
            # discard zero leading term
            while True:
                if message_polynomial[0][0] != 0 or message_polynomial[0][1] < ecc_per_block:
                    break
                message_polynomial = message_polynomial[1:]
        # pad with zeros to ecc_per_block
        return [0] * (ecc_per_block - len(message_polynomial)) + [x[0] for x in message_polynomial]

    blocks_ecc = []
    for _ in range(blocks_in_group_1):
        sub_data = data[:codewords_in_group_1]
        sub_data_ecc = calc_ecc(sub_data)
        blocks_ecc.append(sub_data_ecc)
        data = data[codewords_in_group_1:]
    for _ in range(blocks_in_group_2):
        sub_data = data[:codewords_in_group_2]
        sub_data_ecc = calc_ecc(sub_data)
        blocks_ecc.append(sub_data_ecc)
        data = data[codewords_in_group_2:]
    result_ecc = []
    for i in range(ecc_per_block):
        for block in blocks_ecc:
            if i < len(block):
                result_ecc.append(block[i])
    return result_ecc


def get_matrix_with_quiet_zone(matrix, quiet_zone_size=4):
    """ Create new matrix with quiet zone and copy matrix into it """
    if quiet_zone_size < 0:
        raise Exception('Quiet zone size must be positive')
    size = len(matrix)
    res_matrix = []
    for y in range(size + quiet_zone_size * 2):
        line = []
        for x in range(size + quiet_zone_size * 2):
            line.append(' ')
        res_matrix.append(line)
    # copy matrix to new matrix with quiet zone
    for y in range(len(matrix)):
        for x in range(len(matrix[0])):
            res_matrix[quiet_zone_size + y][quiet_zone_size + x] = matrix[y][x]
    return res_matrix


class QRCode(object):
    """
    QR code
    """

    def __init__(self, error_correction_level, mode_name, mask_number, data=''):
        """
        Create QR code
        """
        self.error_correction_level = error_correction_level
        self.mode_name = mode_name
        self.mask_number = mask_number
        self.data = data
        self.__validate_mode()
        self.size = self.__find_suitable_matrix_size()
        self.matrix = self.__create_matrix(self.size)
        self.__reserved_matrix = self.__create_reserved_matrix(self.size)
        # init variables used for zig zag add data
        self.pos_x = self.size - 1
        self.pos_y = self.size - 1
        self.cur_border_x = self.size - 1
        self.direction = 'up'

    def save(self, filename, module_size=1):
        """ Save QR code to filename """
        self.__encode_data()
        matrix = get_matrix_with_quiet_zone(self.matrix)
        size = len(matrix)
        image = Image.new('RGB', (size * module_size, size * module_size), 'white')
        image_draw = ImageDraw.Draw(image)
        for y in range(len(matrix)):
            for x in range(len(matrix[0])):
                image_draw.rectangle(
                    (x * module_size, y * module_size, x * module_size + module_size, y * module_size + module_size),
                    fill=(0, 0, 0) if matrix[y][x] == '#' else (255, 255, 255))
        image.save(filename)

    @property
    def version(self):
        """ Get QR code version """
        return (len(self.matrix) - 17) / 4

    def print_out(self):
        """ Print out QR code matrix """
        for line in self.matrix:
            print(''.join(x for x in line))

    def __find_suitable_matrix_size(self):
        """ Find suitable matrix size"""
        for v in range(1, 41):
            if capacity.get(v, {}).get(self.error_correction_level, {}).get(self.mode_name) >= len(self.data):
                return 17 + v * 4
        raise Exception('Data too big to fit in one QR code!')

    def __create_matrix(self, size):
        """ Create QR code matrix """
        if (size - 21) % 4 != 0 or size < 21:
            raise Exception('Invalid QR code size! Must be [21, 25, ..., 177]')
        matrix = []
        for _ in range(size):
            line = []
            for __ in range(size):
                line.append(' ')
            matrix.append(line)
        return matrix

    def __create_reserved_matrix(self, size):
        """ Create and fill reserved matrix """
        reserved_matrix = self.__create_matrix(size)
        # add finder patterns
        for y in range(9):
            for x in range(9):
                reserved_matrix[y][x] = '#'
                reserved_matrix[y][size - 8 + x % 8] = '#'
                reserved_matrix[size - 8 + y % 8][x] = '#'
        # add timing patterns
        for i in range(8, size - 8, 1):
            reserved_matrix[6][i] = '#'
            reserved_matrix[i][6] = '#'
        if size < 25:
            return reserved_matrix
        index = (size - 25) / 4
        for pos in itertools.product(alignment_pattern_locations[index], repeat=2):
            for y in range(5):
                for x in range(5):
                    # skip alignment patterns near finder patterns
                    if (pos[0] == 6 and pos[1] == 6) or (pos[0] == 6 and pos[1] == size - 7) or \
                            (pos[0] == size - 7 and pos[1] == 6):
                        continue
                    reserved_matrix[pos[1] - 2 + y][pos[0] - 2 + x] = '#'
        # add version info areas for QR codes version >= 7
        version = (len(reserved_matrix) - 17) / 4
        if version >= 7:
            for i in range(6):
                for j in range(3):
                    reserved_matrix[i][size - 8 - 3 + j] = '#'
                    reserved_matrix[size - 8 - 3 + j][i] = '#'
        return reserved_matrix

    def __add_finder_patterns(self):
        """ Add finder patterns """
        pattern = [
            '#######',
            '#     #',
            '# ### #',
            '# ### #',
            '# ### #',
            '#     #',
            '#######',
        ]
        size = len(self.matrix)
        for y in range(len(pattern)):
            for x in range(len(pattern[0])):
                self.matrix[y][x] = pattern[x][y]
                self.matrix[y][size - len(pattern[0]) + x] = pattern[x][y]
                self.matrix[size - len(pattern) + y][x] = pattern[x][y]

    def __add_alignment_patterns(self):
        """ Add alignment patterns """
        size = len(self.matrix)
        if size < 25:
            return
        pattern = [
            '#####',
            '#   #',
            '# # #',
            '#   #',
            '#####',
        ]
        index = (size - 25) / 4
        for pos in itertools.product(alignment_pattern_locations[index], repeat=2):
            for y in range(len(pattern)):
                for x in range(len(pattern[0])):
                    # skip alignment patterns near finder patterns
                    if (pos[0] == 6 and pos[1] == 6) or (pos[0] == 6 and pos[1] == size - 7) or \
                            (pos[0] == size - 7 and pos[1] == 6):
                        continue
                    self.matrix[pos[1] - len(pattern) / 2 + y][pos[0] - len(pattern[0]) / 2 + x] = pattern[x][y]

    def __add_timing_patterns(self):
        """ Add timing patterns """
        size = len(self.matrix)
        for i in range(8, size - 8, 2):
            self.matrix[6][i] = '#'
            self.matrix[i][6] = '#'

    def __add_format_info(self):
        """ Add format info """
        size = len(self.matrix)
        val = error_correction_levels[self.error_correction_level]
        format_string = bin(val)[2:].zfill(2) + bin(self.mask_number)[2:].zfill(3)
        padded_format_string = format_string.ljust(15, '0').lstrip('0')
        generator_polynomial = bin(0b10100110111)[2:].zfill(11)
        error_correction_string = padded_format_string
        if not error_correction_string:
            error_correction_string = '0'
        while True:
            padded_generator_polynomial = generator_polynomial.ljust(len(error_correction_string), '0')
            out = int(error_correction_string, 2) ^ int(padded_generator_polynomial, 2)
            error_correction_string = bin(out)[2:]
            if len(error_correction_string) <= 10:
                error_correction_string = error_correction_string.zfill(10)
                break
        combined = format_string + error_correction_string
        mask = 0b101010000010010
        mask_string = bin(mask)[2:].zfill(15)
        final_format_string = bin(int(combined, 2) ^ int(mask_string, 2))[2:].zfill(15)
        # add format info to the QR code matrix
        index = 0
        for i in range(9):
            if i == 6:
                continue
            self.matrix[8][i] = '#' if final_format_string[index] == '1' else ' '
            self.matrix[i][8] = '#' if final_format_string[14 - index] == '1' else ' '
            index += 1
        for i in range(8):
            if i < 7:
                self.matrix[size - 1 - i][8] = '#' if final_format_string[i] == '1' else ' '
            self.matrix[8][size - 1 - i] = '#' if final_format_string[14 - i] == '1' else ' '
        self.matrix[size - 8][8] = '#'

    def __add_version_info(self):
        """ Add version patterns """
        size = len(self.matrix)
        # add version info areas for QR codes version >= 7
        version = self.version
        if version < 7:
            return
        # calculate version info
        generator_polynomial = bin(0b1111100100101)[2:].zfill(13)
        version_string = bin(version)[2:].zfill(6)
        padded_version_string = version_string.ljust(18, '0').lstrip('0')
        error_correction_string = padded_version_string
        while True:
            padded_generator_polynomial = generator_polynomial.ljust(len(error_correction_string), '0')
            out = int(error_correction_string, 2) ^ int(padded_generator_polynomial, 2)
            error_correction_string = bin(out)[2:]
            if len(error_correction_string) <= 12:
                error_correction_string = error_correction_string.zfill(12)
                break
        final_version_string = version_string + error_correction_string
        # add the final version string to the QR code
        for i in range(6):
            for j in range(3):
                bit = final_version_string[len(final_version_string) - 1 - (i * 3 + j)]
                self.matrix[i][size - 8 - 3 + j] = '#' if bit == '1' else ' '
                self.matrix[size - 8 - 3 + j][i] = '#' if bit == '1' else ' '

    def __validate_mode(self):
        """ Validate mode """
        if self.mode_name not in modes.keys():
            raise Exception('Invalid mode name. Available modes are: %s.' % ', '.join(sorted(modes, key=modes.get)))

    def __validate_data(self):
        """ Validate data characters """
        if self.mode_name == 'numeric':
            if not self.data.isdigit():
                raise Exception('Numeric mode support only 0..9 characters!')
        elif self.mode_name == 'alphanumeric':
            for char in self.data:
                if char not in alphanumeric_table.keys():
                    raise Exception(
                        'Alphanumeric mode support only alphanumeric characters "%s"!' %
                        ''.join(sorted(alphanumeric_table.keys())))
        elif self.mode_name == 'byte':
            pass
        elif self.mode_name == 'kanji':
            raise Exception('TODO: Kanji mode')

    def __get_data_bits_count(self, mode_bits, data_size_bits):
        """ Get data bits count """
        if self.mode_name == 'numeric':
            remainder_bits = 0
            if len(self.data) % 3 == 0:
                remainder_bits = 0
            elif len(self.data) % 3 == 1:
                remainder_bits = 4
            elif len(self.data) % 3 == 2:
                remainder_bits = 7
            return mode_bits + data_size_bits + 10 * (len(self.data) // 3) + remainder_bits
        elif self.mode_name == 'alphanumeric':
            return mode_bits + data_size_bits + 11 * (len(self.data) // 2) + 6 * (len(self.data) // 2)
        elif self.mode_name == 'byte':
            return mode_bits + data_size_bits + 8 * len(self.data)
        elif self.mode_name == 'kanji':
            return mode_bits + data_size_bits + 13 * len(self.data)
        return 0

    def __add_char(self, char, bits_num):
        """ Add char to QR code matrix """

        def swap_bit(bit):
            """ Swap bit """
            if bit == '1':
                return '0'
            elif bit == '0':
                return '1'
            raise Exception('Bit value must be 1 or 0')

        def apply_mask(bit, x, y, mask_number):
            """ Apply mask pattern to the bit, based on its position """
            if mask_number == 0:
                return swap_bit(bit) if (y + x) % 2 == 0 else bit
            elif mask_number == 1:
                return swap_bit(bit) if y % 2 == 0 else bit
            elif mask_number == 2:
                return swap_bit(bit) if x % 3 == 0 else bit
            elif mask_number == 3:
                return swap_bit(bit) if (y + x) % 3 == 0 else bit
            elif mask_number == 4:
                return swap_bit(bit) if (int(y // 2) + int(x // 3)) % 2 == 0 else bit
            elif mask_number == 5:
                return swap_bit(bit) if ((y * x) % 2) + ((y * x) % 3) == 0 else bit
            elif mask_number == 6:
                return swap_bit(bit) if (((y * x) % 2) + ((y * x) % 3)) % 2 == 0 else bit
            elif mask_number == 7:
                return swap_bit(bit) if (((y + x) % 2) + ((y * x) % 3)) % 2 == 0 else bit
            raise Exception('Data mask pattern must be in range [0..7]')

        def move_up():
            """ Move zig zag up """
            # when left border is riched - move up
            if self.pos_x == self.cur_border_x - 1:
                # if first row - move border and change self.direction to down
                if self.pos_y == 0:
                    # if next module is unused - move there
                    if self.__reserved_matrix[self.pos_y][self.pos_x - 1] != '#':
                        self.pos_x -= 1
                    else:
                        # find unused module to the top
                        found = False
                        for py in range(0, self.size, 1):
                            if self.__reserved_matrix[py][self.pos_x - 1] != '#':
                                found = True
                                self.pos_y = py
                                self.pos_x -= 1
                                break
                        if not found:
                            raise Exception('Unused module to the top was not found!')
                    self.cur_border_x -= 2
                    self.direction = 'down'
                else:
                    # if next module is unused - move there
                    if self.__reserved_matrix[self.pos_y - 1][self.pos_x + 1] != '#':
                        self.pos_x += 1
                        self.pos_y -= 1
                    elif self.__reserved_matrix[self.pos_y - 1][self.pos_x] != '#':
                        # if next module is unused - move there
                        self.pos_y -= 1
                    else:
                        # if next module is already used or reserved, check modules to the top
                        found = False
                        found_x_1_y = self.pos_y
                        found_x_0_y = self.pos_y
                        for py in range(self.pos_y - 1, -1, -1):
                            if self.__reserved_matrix[py][self.pos_x + 1] != '#':
                                found = True
                                found_x_1_y = py
                                break
                        # try self.pos_x
                        for py in range(self.pos_y - 1, -1, -1):
                            if self.__reserved_matrix[py][self.pos_x] != '#':
                                found = True
                                found_x_0_y = py
                                break
                        # get nearest found self.pos_y
                        if found:
                            if found_x_1_y >= found_x_0_y:
                                self.pos_y = found_x_1_y
                                self.pos_x += 1
                            else:
                                self.pos_y = found_x_0_y
                        else:
                            # if the top right version block is reached
                            if self.pos_x == self.size - 10:
                                # try self.pos_x - 1
                                if not found:
                                    for py in range(self.pos_y - 1, -1, -1):
                                        if self.__reserved_matrix[py][self.pos_x - 1] != '#':
                                            found = True
                                            self.pos_x -= 1
                                            break
                                # try self.pos_x - 2
                                if not found:
                                    for py in range(self.pos_y - 1, -1, -1):
                                        if self.__reserved_matrix[py][self.pos_x - 2] != '#':
                                            found = True
                                            self.pos_x -= 2
                                            break
                                if found:
                                    for py in range(self.pos_y - 1, -1, -1):
                                        if py == 6:
                                            continue
                                        if self.__reserved_matrix[py][self.pos_x] != '#':
                                            self.pos_y = py
                                        else:
                                            break
                                self.cur_border_x -= 2
                            # if left timing is reached - skip it, move border and change self.direction to down
                            elif self.pos_x == 7:
                                # skip - move one more left
                                self.pos_x -= 2
                                self.cur_border_x -= 3
                            else:
                                # else just move border and change self.direction to down
                                self.pos_x -= 1
                                self.cur_border_x -= 2
                            self.direction = 'down'
            else:
                # move left
                self.pos_x -= 1

        def move_down():
            """ Move zig zag down """
            # when left border is riched - move down
            if self.pos_x == self.cur_border_x - 1:
                # if last row - move border and change self.direction to up
                if self.pos_y == self.size - 1:
                    # if next module is unused - move there
                    if self.__reserved_matrix[self.pos_y][self.pos_x - 1] != '#':
                        self.pos_x -= 1
                    else:
                        # else find unused module to the top
                        found = False
                        for py in range(self.pos_y - 1, -1, -1):
                            if self.__reserved_matrix[py][self.pos_x - 1] != '#':
                                found = True
                                self.pos_y = py
                                self.pos_x -= 1
                                break
                        if not found:
                            raise Exception('Unused module to the top was not found!')
                    self.cur_border_x -= 2
                    self.direction = 'up'
                else:
                    # if next module is unused - move there
                    if self.__reserved_matrix[self.pos_y + 1][self.pos_x + 1] != '#':
                        self.pos_x += 1
                        self.pos_y += 1
                    elif self.__reserved_matrix[self.pos_y + 1][self.pos_x] != '#':
                        # if next module is unused - move there
                        self.pos_y += 1
                    else:
                        # if next module is already used or reserved, check modules to the bottom
                        found = False
                        found_x_1_y = self.pos_y
                        found_x_0_y = self.pos_y
                        for py in range(self.pos_y + 1, self.size, 1):
                            if self.__reserved_matrix[py][self.pos_x + 1] != '#':
                                found = True
                                found_x_1_y = py
                                break
                        for py in range(self.pos_y + 1, self.size, 1):
                            if self.__reserved_matrix[py][self.pos_x] != '#':
                                found = True
                                found_x_0_y = py
                                break
                        # get nearest found self.pos_y
                        if found:
                            if found_x_1_y <= found_x_0_y:
                                self.pos_y = found_x_1_y
                                self.pos_x += 1
                            else:
                                self.pos_y = found_x_0_y
                        else:
                            self.pos_x -= 1
                            self.cur_border_x -= 2
                            self.direction = 'up'
            else:
                # move left
                self.pos_x -= 1

        # add char to the matrix bit by bit
        bits = list(bin(ord(char))[2:].zfill(bits_num))
        while bits:
            bit_str = bits.pop(0)
            bit_str = apply_mask(bit_str, self.pos_x, self.pos_y, self.mask_number)
            self.matrix[self.pos_y][self.pos_x] = '#' if bit_str == '1' else ' '
            if self.direction == 'up':
                move_up()
            elif self.direction == 'down':
                move_down()

    def __add_bytes_data(self):
        """ Add data to QR code matrix """
        self.__validate_data()
        full_data_bits = ''

        # add mode
        mode_bits = 4
        mode = modes.get(self.mode_name)
        full_data_bits += bin(mode)[2:].zfill(mode_bits)

        # add data size (character count)
        data_size_bits = get_character_count_bits(self.version, self.mode_name)
        data_size = len(self.data)
        full_data_bits += bin(data_size)[2:].zfill(data_size_bits)

        # add data chars
        bits_per_char = 8
        if self.mode_name == 'numeric':
            full_data_bits += numeric_to_binary(self.data)
        elif self.mode_name == 'alphanumeric':
            full_data_bits += alphanumeric_to_binary(self.data)
        elif self.mode_name == 'byte':
            for char in self.data:
                full_data_bits += bin(ord(char))[2:].zfill(bits_per_char)
        elif self.mode_name == 'kanji':
            raise Exception('TODO: Kanji mode')

        # add terminator
        term_size = term_sizes.get('all', 4)
        full_data_bits += bin(0b0)[2:].zfill(term_size)

        # align to 8 bits
        if len(full_data_bits) % 8:
            full_data_bits += bin(0b0)[2:].zfill(8 - len(full_data_bits) % 8)

        # add padding
        padding_bits = 8
        padding_index = 0
        padding_count = capacity.get(
            self.version, {}).get(self.error_correction_level, {}).get('bits', 0) / 8 - (len(full_data_bits) / 8)
        while True:
            if padding_index >= padding_count:
                break
            full_data_bits += bin(0b11101100)[2:].zfill(padding_bits)  # 0xec
            padding_index += 1
            if padding_index >= padding_count:
                break
            full_data_bits += bin(0b00010001)[2:].zfill(padding_bits)  # 0x11
            padding_index += 1

        # convert bits to bytes
        data_bytes = [int(full_data_bits[i: i + 8], 2) for i in range(0, len(full_data_bits), 8)]

        # prepare result data after interleaving group's blocks
        error_correction_info = error_correction.get(self.version, {}).get(self.error_correction_level, {})
        blocks_in_group_1 = error_correction_info.get('blocks_in_group_1')
        blocks_in_group_2 = error_correction_info.get('blocks_in_group_2')
        codewords_in_group_1 = error_correction_info.get('codewords_in_group_1')
        codewords_in_group_2 = error_correction_info.get('codewords_in_group_2')
        data_blocks = []
        # group 1
        for _ in range(blocks_in_group_1):
            sub_data_bytes = data_bytes[:codewords_in_group_1]
            data_blocks.append(sub_data_bytes)
            data_bytes = data_bytes[codewords_in_group_1:]
        # group 2
        for _ in range(blocks_in_group_2):
            sub_data_bytes = data_bytes[:codewords_in_group_2]
            data_blocks.append(sub_data_bytes)
            data_bytes = data_bytes[codewords_in_group_2:]
        result_data = []
        for i in range(max(codewords_in_group_1, codewords_in_group_2)):
            for block in data_blocks:
                if i < len(block):
                    result_data.append(block[i])
        # add result data
        for char in result_data:
            self.__add_char(chr(char), bits_per_char)

        # add error correction
        data_bytes = [int(full_data_bits[i: i + 8], 2) for i in range(0, len(full_data_bits), 8)]
        error_correction_data = get_error_correction_data(data_bytes, self.version, self.error_correction_level)
        for error_correction_byte in error_correction_data:
            self.__add_char(chr(error_correction_byte), bits_per_char)
            full_data_bits += bin(error_correction_byte)[2:].zfill(bits_per_char)

        # skip addition of remainder bits, as all remainder bits are 0

    def __encode_data(self):
        """ Encode data """
        self.__add_finder_patterns()
        self.__add_timing_patterns()
        self.__add_alignment_patterns()
        self.__add_format_info()
        self.__add_version_info()
        self.__add_bytes_data()


def main():
    qr_code = QRCode(error_correction_level='L', mode_name='byte', mask_number=2, data='encoded message')
    qr_code.save('qr.png', module_size=2)
    print('result QR Code version: %d' % qr_code.version)


if __name__ == '__main__':
    main()
