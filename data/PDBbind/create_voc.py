import re
import os

train_file = 'dataPre/PDBbind-train'
test_active = 'active_smile'
test_decoy = 'decoy_smile'
fname = 'voc/combinedVoc-wholeFour.voc'
regex = '(\[[^\[\]]{1,10}\])'
add_chars = set()
smiles_list = []
old_vocab = '''[102Ru]
[80Se]
[N-]
[33SH2]
[Ag+]
L
[Fr]
[C@@H-]
[SiH2]
[Sr+2]
[P@@H]
[CH2]
[10B]
[SH]
[Gd]
[Ru+2]
[81R-]
[Ar]
[85Sr+2]
[29Si]
[Cu+2]
[CH3-]
[56Ni]
[Na+]
[Au+]
[Co+3]
[Se-]
[NH-]
[113In]
)
[Ir]
[NH+]
[109Cd]
[R-]
[43K]
[15NH2]
[Zn+2]
[P@H]
[Ce]
[As+5]
[Pt+4]
n
6
[Ca+2]
[121Sn]
4
[nH+]
R
[F]
[Pd]
3
[R]
\\
#
[H-]
[16OH2]
[O-2]
[nH]
[Ni+2]
[CH]
I
[Ho]
[C@H-]
[209Pb]
p
F
[18F]
[183W]
[34SH2]
[Co+2]
[CH2-]
[Sb]
[Be+2]
[n+]
[13C]
[Fe+2]
[B]
[178W]
[67Ga+3]
[3H]
[63Cu]
[79R-]
[n-]
s
[C+]
[205Pb]
[H+]
[O-]
(
[H]
[OH2+]
[Mn]
[Se]
[78Se]
[s+]
[NH2-]
[Fe]
[C@@]
[OH-]
[S@]
5
[173Ta]
[11CH4]
[L-]
[128IH]
[Mo]
[As+]
[B-]
[129IH]
[Ta+5]
[68Ge]
C
[77RH]
/
[Hg+]
[NH4+]
[P@@+]
[CH-]
[84RH]
[43Ca]
[S-2]
[11CH3]
[Fe+3]
[Ca]
[125Sb]
[Sb-]
[P@]
[210Tl]
[PH+]
[K]
[SH+]
o
[Si+4]
[89Sr+2]
[I]
[15NH3]
O
[14CH2]
[33P]
c
=
[C-]
[127IH]
[141Ce]
[75Se]
8
[11B]
[P@@]
B
.
N
[P@+]
[Hg]
[252Cf]
[Pb+2]
[O+]
[56Fe]
[13CH]
[45Ca+2]
[Zr]
[45K]
[C@H]
[In+3]
[S@@]
[Si]
[99Tc]
[Hg+2]
[o+]
[Ba+2]
[N@+]
1
9
[AlH4-]
[44Ca]
[N+]
[Cu]
[9Li]
[Tl+]
[Co]
[Re]
[OH3+]
[Mn+2]
[Ru+]
[L]
[Li+]
[V]
<pad>
7
[115Cd]
[Mg+2]
[P+]
[Pd+2]
[I-]
[Cr+3]
[Zn]
[C@]
[110Ag]
[214Pb]
2
[60Fe]
S
[NH2+]
[P@@H+]
[W]
[Al+3]
[F-]
[OH]
[125IH]
[N@@+]
[C]
[13NH3]
[Bi]
[218Po]
[S-]
[As]
[SiH3]
[23Na]
[Ge]
[197Au]
[32P]
[12CH4]
[C@@H]
[17NH3]
[NH3+]
[132IH]
[99Mo]
[207Pb]
[7Li]
[2H]
[Al]
[OH+]
[Zr+4]
[S+]
[125I]
-
[Ag]
[Sb+3]
[3He]
[90Sr+2]
[K+]
[Y+3]
[Ti]
[188W]
[Rb+]
[Pt+2]
[14C]
P
[S@H+]
[N@@H+]
[N@H+]
[N-2]
[N+2]
[c]
[Ru]
[B-2]
[PH]
[Mg@@-2]'''

with open(train_file, "r") as f:
    lines = f.read().split()
    for line in lines:
        ligand = line.split(" ")[0]
        smiles_list.append(ligand)

for af in os.listdir(test_active):
    with open(os.path.join(test_active, af), "r") as f:
        lines = f.read().split()
        for line in lines:
            ligand = line.split(" ")[0]
            smiles_list.append(ligand)

for df in os.listdir(test_decoy):
    with open(os.path.join(test_decoy, df), "r") as f:
        lines = f.read().split()
        for line in lines:
            ligand = line.split(" ")[0]
            smiles_list.append(ligand)

def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string

for i, smiles in enumerate(smiles_list):
    regex = '(\[[^\[\]]{1,10}\])'
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    for char in char_list:
        if char.startswith('['):
            add_chars.add(char)
        else:
            chars = [unit for unit in char]
            [add_chars.add(unit) for unit in chars]

old_chars = [x.strip() for x in old_vocab.split("\n")]
add_chars = list(set(old_chars) | set(add_chars))
print("Number of characters: {}".format(len(add_chars)))
with open(fname, 'w') as f:
    f.write('<pad>' + "\n")
    for char in add_chars:
        f.write(char + "\n")