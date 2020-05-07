import torch
import random
import hashlib

def and_bit(a, b):
    return a * b

def xor_bit(a, b):
    return 1.0 - (1.0 - a + a * a * b) * (1.0 - b + a * b * b)

def or_bit(a, b):
    return 1.0 - (1.0 - a * a) * (1.0 - b * b)

def not_bit(a):
    return 1.0 - a

def xor(a, b):
    return [xor_bit(a_bit, b_bit) for (a_bit, b_bit) in zip(a, b)]

def and_(a, b):
    return [and_bit(a_bit, b_bit) for (a_bit, b_bit) in zip(a, b)]

def add(a, b):
    sum_0 = xor_bit(a[-1], b[-1])
    carry = and_bit(a[-1], b[-1])
    def full_adder(a_bit, b_bit, carry_bit):
        r1 = xor_bit(a_bit, b_bit)
        sum = xor_bit(r1, carry_bit)
        r2 = and_bit(r1, carry_bit)
        r3 = and_bit(a_bit, b_bit)
        carry_bit_out = or_bit(r2, r3)
        return (sum, carry_bit_out)

    ret = [sum_0]
    for (a_bit, b_bit) in list(reversed(list(zip(a, b))))[1:]:
        (sum_i, carry) = full_adder(a_bit, b_bit, carry)
        ret.insert(0, sum_i)

    return ret

def not_(a):
    return [not_bit(a_bit) for a_bit in a]

def right_rotate(a, c):
    return a[-c:] + a[0:-c]

def right_shift(a, c):
    return ([0.0] * c) + a[0:-c]

def bits_to_num(a):
    total = a[-1]
    multiplier = 2.0
    for bit in list(reversed(a))[1:]:
        total = total + bit * multiplier
        multiplier *= 2.0
    return total

def left_shift(a, c):
    return a + [0.0] * c

def num_to_8_bits(num):
    s = "{0:{fill}8b}".format(num, fill='0')
    return [1 if c == '1' else 0 for c in s]

def num_to_32_bits(num):
    s = "{0:{fill}32b}".format(num, fill='0')
    return [1 if c == '1' else 0 for c in s]

def num_to_64_bits(num):
    s = "{0:{fill}64b}".format(num, fill='0')
    return [1 if c == '1' else 0 for c in s]

def flatten(lst_of_lsts):
    return [x for lst in lst_of_lsts for x in lst]

# Initialize hash values
h0n = 0x6a09e667
h1n = 0xbb67ae85
h2n = 0x3c6ef372
h3n = 0xa54ff53a
h4n = 0x510e527f
h5n = 0x9b05688c
h6n = 0x1f83d9ab
h7n = 0x5be0cd19

# Initialize array of round constants

kn = [
   0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
   0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
   0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
   0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
   0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
   0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
   0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
   0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]

parallel_tries = 1

parallel_zeros = torch.tensor([0.0]).expand(parallel_tries)
parallel_ones = torch.tensor([1.0]).expand(parallel_tries)

def expand_constant(n):
    return [parallel_zeros if bit == 0 else parallel_ones for bit in n]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def sha256(message):
    h0 = expand_constant(num_to_32_bits(h0n))
    h1 = expand_constant(num_to_32_bits(h1n))
    h2 = expand_constant(num_to_32_bits(h2n))
    h3 = expand_constant(num_to_32_bits(h3n))
    h4 = expand_constant(num_to_32_bits(h4n))
    h5 = expand_constant(num_to_32_bits(h5n))
    h6 = expand_constant(num_to_32_bits(h6n))
    h7 = expand_constant(num_to_32_bits(h7n))

    k = [expand_constant(num_to_32_bits(n)) for n in kn]

    # Pre-processing (Padding)
    # begin with the original message of length L bits
    L = len(message)

    # append a single '1' bit
    message = message + [parallel_ones]
    # append K '0' bits, where K is the minimum number >= 0 such that L + 1 + K + 64 is a multiple of 512
    K = 0
    while True:
        if (L + 1 + K + 64) % 512 == 0:
            break
        K += 1

    message = message + [parallel_zeros] * K

    # append L as a 64-bit big-endian integer, making the total post-processed length a multiple of 512 bits
    message = message + num_to_64_bits(L)

    for chunk in chunks(message, 512):
        w = list(chunks(chunk, 32)) + [None] * (64 - 16)
        for i in range(16, 64):
            # s0 := (w[i-15] rightrotate  7) xor (w[i-15] rightrotate 18) xor (w[i-15] rightshift  3)
            s0 = xor(xor(right_rotate(w[i - 15], 7), right_rotate(w[i - 15], 18)), right_shift(w[i - 15], 3))
            # s1 := (w[i- 2] rightrotate 17) xor (w[i- 2] rightrotate 19) xor (w[i- 2] rightshift 10)
            s1 = xor(xor(right_rotate(w[i - 2], 17), right_rotate(w[i - 2], 19)), right_shift(w[i - 2], 10))
            # w[i] := w[i-16] + s0 + w[i-7] + s1
            w[i] = add(add(add(w[i - 16], s0), w[i - 7]), s1)

        # Initialize working variables to current hash value:
        a = h0
        b = h1
        c = h2
        d = h3
        e = h4
        f = h5
        g = h6
        h = h7

        # Compression function main loop:
        for i in range(0, 64):
            # S1 := (e rightrotate 6) xor (e rightrotate 11) xor (e rightrotate 25)
            S1 = xor(xor(right_rotate(e, 6), right_rotate(e, 11)), right_rotate(e, 25))
            # ch := (e and f) xor ((not e) and g)
            ch = xor(and_(e, f), and_(not_(e), g))
            # temp1 := h + S1 + ch + k[i] + w[i]
            temp1 = add(add(add(add(h, S1), ch), k[i]), w[i])
            # S0 := (a rightrotate 2) xor (a rightrotate 13) xor (a rightrotate 22)
            S0 = xor(xor(right_rotate(a, 2), right_rotate(a, 13)), right_rotate(a, 22))
            # maj := (a and b) xor (a and c) xor (b and c)
            maj = xor(xor(and_(a, b), and_(a, c)), and_(b, c))
            # temp2 := S0 + maj
            temp2 = add(S0, maj)

            h = g
            g = f
            f = e
            e = add(d, temp1)
            d = c
            c = b
            b = a
            a = add(temp1, temp2)

        # Add the compressed chunk to the current hash value:
        h0 = add(h0, a)
        h1 = add(h1, b)
        h2 = add(h2, c)
        h3 = add(h3, d)
        h4 = add(h4, e)
        h5 = add(h5, f)
        h6 = add(h6, g)
        h7 = add(h7, h)

    return h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7

hello_world = "Hello world"
hello_world_hash = 0x64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c
hello_world_ascii = [ord(c) for c in hello_world]
hello_world_binary = flatten([num_to_8_bits(c) for c in hello_world_ascii])
assert(len(hello_world_binary) % 8 == 0)
hello_world_torch = expand_constant(hello_world_binary)
hello_world_digest = sha256(hello_world_torch)

bits = [round(float(bit[0])) for bit in hello_world_digest]
bits_string = ''.join(['0' if b == 0 else '1' for b in bits])
digest_as_int = int(bits_string, 2)
digest_as_hex = hex(digest_as_int)

assert(digest_as_int == hello_world_hash)

leading_zeros = 32

hw_leading_digest = hello_world_digest[0:leading_zeros]
hw_leading_float = float(bits_to_num(hw_leading_digest)[0])

#raw_nonce = [torch.randn(parallel_tries, requires_grad=True) for _ in range(32)]
#for n in raw_nonce:
    #n[0] = 0.5

raw_nonce = [torch.zeros(parallel_tries, requires_grad=True) for _ in range(32)]

prev_hash_py = [random.choice([0, 1]) for _ in range(256)]
prev_hash = expand_constant(prev_hash_py)

optimizer = torch.optim.SGD(raw_nonce, lr=10.0, momentum=0.9)

def normalize(input, dim):
    norm = torch.norm(input, dim=dim, keepdim=True)
    norm_expanded = norm.expand_as(input)
    return input / norm_expanded

while True:
    nonce = [torch.sigmoid(bit) for bit in raw_nonce]
    message0 = nonce + prev_hash

    optimizer.zero_grad()
    #digest = sha256(sha256(message0))
    digest = sha256(message0)
    leading_digest = digest[0:leading_zeros]
    leading_float = bits_to_num(leading_digest)
    total = torch.sum(leading_float)
    total.backward()

    print([r.grad for r in raw_nonce])
    print(total)
    print(leading_float)
    print(nonce)

    #optimizer.step()

    # Normalize the gradient
    stacked_grad = torch.stack([r.grad for r in raw_nonce])
    normalized_stacked_grad = normalize(stacked_grad, dim=0)

    for (i, raw_nonce_bit) in enumerate(raw_nonce):
        raw_nonce_bit.data -= normalized_stacked_grad[i]

    if input("Stop computation?") == "y":
        break

def bitstring_to_bytes(s):
    return int(s, 2).to_bytes(len(s) // 8, byteorder='big')

nonce = [torch.sigmoid(bit) for bit in raw_nonce]

for i in range(parallel_tries):
    bits_py = []
    for bit in nonce:
        b = round(float(bit[i]))
        bits_py.append(b)
    computed_message = bits_py + prev_hash_py
    computed_message_str = ''.join([str(bit) for bit in computed_message])
    message_bytes = bitstring_to_bytes(computed_message_str)
    print("messsage_bytes:")
    print(message_bytes)
    print("sha256 digest:")
    print(hashlib.sha256(message_bytes).hexdigest())
