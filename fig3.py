import numpy as np
import matplotlib.pyplot as plt

E = {
    1: 0.5,
    2: 2.86153,
    3: 7.43277,
    4: 14.5732,
    5: 24.5296,
    6: 37.6596,
    7: 54.2973,
    8: 74.7737,
    9: 99.4107,    
    10: 128.542,
    11: 161.852,
    12: 199.615,
    13: 241.877,
    14: 288.878,
    15: 340.649,
    16: 397.480,
    17: 459.480,
    18: 526.860,
    19: 599.121,
    20: 676.721,
    21: 759.729,
    22: 848.379,
    23: 942.883,
    25: 1149.66,
    26: 1262.27,
    27: 1381.27,
    31: 1923.26,
    32: 2075.28,
    33: 2234.22,
    34: 2399.78,
    35: 2572.29,
    36: 2752.19,
    37: 2938.47,
    38: 3131.54,
    39: 3331.55,
    40: 3538.80,
    41: 3753.30,
    42: 3975.20,
    43: 4204.44,
    44: 4441.27,
    45: 4685.64,
    48: 5465.04,
    49: 5740.00,
    50: 6022.77,
    51: 6313.50,
    52: 6612.13,
    53: 6918.17,
    55: 7553.86,
    56: 7884.05,
    57: 8221.48,
    59: 8921.5,
    60: 9283.82,
    62: 10034.9,
    63: 10423.1,
    74: 15287.4,
    75: 15784.3,
    76: 16290.4,
    77: 16805.7,
    78: 17330.7,
    80: 18408.6,
    81: 18961.5,
    82: 19523.6,
    83: 20095.5,
    84: 20676.4,
    85: 21266.8,
    87: 22475.7,
    89: 23722.1,
    90: 24360.5,
    92: 25663.1,
}

# data from:
#   https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html
E_exp = [
    13.59843449,
    79.0051538,
    203.4861694,
    399.14864,
    670.9809,
    1030.1084,
    1486.058,
    2043.8428,
    2715.890,
    3511.696,
    4420.00,
    5451.06,
    6604.95,
    7888.53,
    9305.8,
    10859.7,
    12556.4,
    14400.8,
    16382,
    18510,
    20788,
    23221,
    25820,
    28582,
    31514,
    34619,
    37899,
    41356,
    45000,
    48840,
    52870,
    57080,
    61470,
    66070,
    70870,
    75870,
    81050,
    86440,
    92030,
    97820,
    103840,
    110060,
    116480,
    123140,
    130030,
    137140,
    144480,
    152050,
    159850,
    167890,
    176140,
    184650,
    193400,
    202400,
    211630,
    221100,
    230820,
    240800,
    251060,
    261580,
    272380,
    283450,
    294810,
    306410,
    318330,
    330600,
    343100,
    355800,
    369000,
    382500,
    396300,
    410300,
    424500,
    438990,
    454200,
    469400,
    485000,
    500900,
    517100,
    533900,
    550900,
    568200,
    585800,
    603900,
    622300,
    641000,
    660200,
    680000,
    700000,
    720000,
    740000,
    761000,
]     

Z = np.arange(len(E_exp))+1
plt.semilogy(Z,E_exp,label='experiment')

Z = list(E.keys())
E = np.array(list(E.values()))*2*E_exp[0]
plt.semilogy(Z,E,'+',label='calculation')

plt.xlabel('Z = atomic nubmer', fontsize=14)
plt.ylabel('total binding energy  / eV', fontsize=14)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('fig3.eps')
plt.show()