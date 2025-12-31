# Adapted from https://github.com/basalt-org/basalt/blob/076de80812dde323d9de63ea8975ad1875243dc0/basalt/utils/uuid.mojo

from time import perf_counter_ns
from utils import StaticTuple
from hashlib.hasher import Hasher


@register_passable("trivial")
struct MersenneTwister:
    """
    Pseudo-random generator Mersenne Twister (MT19937-32bit).
    """

    comptime N: Int = 624
    comptime M: Int = 397
    comptime MATRIX_A: Int32 = 0x9908B0DF
    comptime UPPER_MASK: Int32 = 0x80000000
    comptime LOWER_MASK: Int32 = 0x7FFFFFFF
    comptime TEMPERING_MASK_B: Int32 = 0x9D2C5680
    comptime TEMPERING_MASK_C: Int32 = 0xEFC60000

    var state: StaticTuple[Int32, Self.N]
    var index: Int

    fn __init__(out self, seed: Int):
        comptime W: Int = 32
        comptime F: Int32 = 1812433253
        comptime D: Int32 = 0xFFFFFFFF

        self.index = Self.N
        self.state = StaticTuple[Int32, Self.N]()
        self.state[0] = seed & D

        for i in range(1, Self.N):
            self.state[i] = (
                F * (self.state[i - 1] ^ (self.state[i - 1] >> (W - 2))) + i
            ) & D

    fn next(mut self) -> Int32:
        if self.index >= Self.N:
            for i in range(Self.N):
                var x = (self.state[i] & Self.UPPER_MASK) + (
                    self.state[(i + 1) % Self.N] & Self.LOWER_MASK
                )
                var xA = x >> 1
                if x % 2 != 0:
                    xA ^= Self.MATRIX_A
                self.state[i] = self.state[(i + Self.M) % Self.N] ^ xA
            self.index = 0

        var y = self.state[self.index]
        y ^= y >> 11
        y ^= (y << 7) & Self.TEMPERING_MASK_B
        y ^= (y << 15) & Self.TEMPERING_MASK_C
        y ^= y >> 18
        self.index += 1

        return y

    fn next_ui8(mut self) -> UInt8:
        return UInt8(self.next()) & 0xFF


@register_passable("trivial")
struct UUID(Copyable, Equatable, Hashable, Movable, Stringable, Writable):
    var bytes: StaticTuple[UInt8, 16]

    fn __init__(out self):
        self.bytes = StaticTuple[UInt8, 16]()

    fn __setitem__(mut self, index: Int, value: UInt8):
        self.bytes[index] = value

    fn __getitem__(self, index: Int) -> UInt8:
        return self.bytes[index]

    fn __eq__(self, other: Self) -> Bool:
        @parameter
        for i in range(16):
            if self.bytes[i] != other.bytes[i]:
                return False
        return True

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __hash__[H: Hasher](self, mut hasher: H):
        # Hash all bytes
        @parameter
        for i in range(16):
            hasher.update(self.bytes[i])

    fn __str__(self) -> String:
        var result: String = ""
        comptime hex_digits: String = "0123456789abcdef"

        @parameter
        for i in range(16):
            if i == 4 or i == 6 or i == 8 or i == 10:
                result += "-"
            result += (
                hex_digits[Int(self.bytes[i] >> 4)]
                + hex_digits[Int(self.bytes[i] & 0xF)]
            )
        return result

    fn write_to(self, mut writer: Some[Writer]) -> None:
        writer.write(String(self))


@register_passable("trivial")
struct UUIDGenerator:
    var prng: MersenneTwister

    fn __init__(out self, seed: Int):
        self.prng = MersenneTwister(seed)

    fn next(mut self) -> UUID:
        var uuid = UUID()

        @parameter
        for i in range(16):
            uuid[i] = self.prng.next_ui8()

        # Version 4, variant 10xx
        uuid[6] = 0x40 | (0x0F & uuid[6])
        uuid[8] = 0x80 | (0x3F & uuid[8])

        return uuid


fn generate_uuid(seed: Optional[Int] = None) -> UUID:
    """Generate a new UUID using a Mersenne Twister PRNG seeded with the given seed.

    Args:
        seed: An integer seed for the PRNG.

    Returns:
        A newly generated UUID.
    """
    s = seed.value() if seed else Int(perf_counter_ns())
    var generator = UUIDGenerator(s)
    return generator.next()
